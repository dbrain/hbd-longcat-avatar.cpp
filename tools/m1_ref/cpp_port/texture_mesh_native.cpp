// texture_mesh_native -- native TRELLIS-2 texture-only pass for an existing mesh.
//
// This is the production counterpart of texture_mesh.py.  It deliberately starts with the
// FINAL (e.g. UltraShape-refined) mesh instead of rebaking the geometry pass's low-frequency PBR
// volume.  The chain is:
//   mesh -> Trellis2 canonical frame -> flexible dual grid -> shape SLat encoder
//        -> DINOv3 image conditioning -> cross texture SLat DiT -> PBR decoder -> UV bake.
// No Python is involved at run time.
#include "pixal3d_chain.hpp"
#include "voxelizer.hpp"
#include "shape_slat_encoder.hpp"
#include "glb_reader.hpp"
#include "glb_textured.hpp"
#include "tex_atlas.hpp"
#include "image_io.hpp"
#include "../../sparse_spike/npy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static void usage() {
    std::printf("usage: texture_mesh_native --model <geo_gguf_dir> --mesh <in.glb> --image <matte.png> --out <out.glb>\n"
                "                           [--resolution 512|1024] [--texsize N] [--seed N] [--decimate F] [--dump-dir DIR]\n"
                "                           [--unwrap production|reference]\n");
}

template<class T> static void dump_npy(const std::string& path, const std::vector<T>& data,
                                       const char* descr, const std::vector<int>& shape) {
    std::string dims="(";
    for (size_t i=0;i<shape.size();i++) { if (i) dims += ", "; dims += std::to_string(shape[i]); }
    if (shape.size()==1) dims += ",";
    dims += ")";
    std::string h="{'descr': '"+std::string(descr)+"', 'fortran_order': False, 'shape': "+dims+", }";
    const size_t pad=(16-((10+h.size()+1)%16))%16; h.append(pad,' '); h.push_back('\n');
    FILE* f=std::fopen(path.c_str(),"wb");
    if (!f) throw std::runtime_error("cannot write dump: "+path);
    std::fwrite("\x93NUMPY\x01\x00",1,8,f); const uint16_t n=(uint16_t)h.size(); std::fwrite(&n,2,1,f);
    std::fwrite(h.data(),1,h.size(),f); std::fwrite(data.data(),sizeof(T),data.size(),f); std::fclose(f);
}

static std::vector<float> load_norm(const std::string& model, const char* which) {
    for (const std::string& p : {model + "/" + which + ".npy", std::string("refs/") + which + ".npy"}) {
        FILE* f = std::fopen(p.c_str(), "rb");
        if (f) { std::fclose(f); NpyArray a = npy_load(p); return {a.f32(), a.f32() + a.numel()}; }
    }
    throw std::runtime_error(std::string("missing normalization: ") + which);
}

// Match Trellis2TexturingPipeline.preprocess_mesh exactly.  The swap is undone only after baking,
// so the output remains in the original GLB coordinate convention.
static void preprocess_mesh(std::vector<float>& v) {
    float mn[3] = {1e30f,1e30f,1e30f}, mx[3] = {-1e30f,-1e30f,-1e30f};
    for (size_t i=0;i<v.size();i+=3) for (int c=0;c<3;c++) {
        mn[c] = std::min(mn[c], v[i+c]); mx[c] = std::max(mx[c], v[i+c]);
    }
    float center[3], extent=0.f;
    for (int c=0;c<3;c++) { center[c]=0.5f*(mn[c]+mx[c]); extent=std::max(extent,mx[c]-mn[c]); }
    const float scale = 0.99999f / std::max(extent, 1e-12f);
    for (size_t i=0;i<v.size();i+=3) {
        float x=(v[i]-center[0])*scale, y=(v[i+1]-center[1])*scale, z=(v[i+2]-center[2])*scale;
        v[i]=x; v[i+1]=-z; v[i+2]=y;
    }
}

static void restore_mesh_frame(std::vector<float>& v) {
    for (size_t i=0;i<v.size();i+=3) { float y=v[i+1], z=v[i+2]; v[i+1]=z; v[i+2]=-y; }
}

// Trellis2TexturingPipeline receives a PIL image, optionally crops its transparent subject, then
// DinoV3FeatureExtractor does the final square Lanczos resize.  The runbook inputs are already
// matted; this keeps their transparent padding from changing the image conditioning.
static imgio::Image load_texture_image(const std::string& path) {
    int w=0,h=0,c=0;
    unsigned char* p=stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!p) throw std::runtime_error("cannot read image: " + path);
    imgio::Image im; im.w=w; im.h=h; im.rgb.resize((size_t)w*h*3);
    std::vector<float> alpha01((size_t)w*h);
    for (int y=0;y<h;y++) for (int x=0;x<w;x++) for (int ch=0;ch<3;ch++) {
        im.rgb[((size_t)y*w+x)*3+ch]=p[((size_t)y*w+x)*4+ch]/255.f;
        alpha01[(size_t)y*w+x]=p[((size_t)y*w+x)*4+3]/255.f;
    }
    bool has_alpha=false;
    for (int i=0;i<w*h;i++) if (p[i*4+3] != 255) { has_alpha=true; break; }
    // The runbook's common "matte" input is opaque RGB with a black background.  Python routes
    // that through RMBG before it crops; until the learned matte model is ported, recover this
    // lossless case natively instead of conditioning DINO on a large black border.  Only enable it
    // when the border is overwhelmingly near-black, so ordinary dark photographs are untouched.
    if (!has_alpha) {
        int border=0, black=0;
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) if (x==0||y==0||x==w-1||y==h-1) {
            border++; const float* q=&im.rgb[((size_t)y*w+x)*3];
            if (std::max({q[0],q[1],q[2]}) < .035f) black++;
        }
        if (border && black > border*95/100) {
            for (size_t i=0;i<alpha01.size();i++) {
                const float* q=&im.rgb[i*3]; float m=std::max({q[0],q[1],q[2]});
                alpha01[i]=std::max(0.f,std::min(1.f,(m-.01f)/.06f));
            }
            has_alpha=true;
        }
    }
    // preprocess_image downsizes BEFORE finding the alpha bbox.  Resize alpha through the same
    // Lanczos kernel (as three identical channels) so the subsequent threshold/crop matches PIL.
    const int max_side=std::max(w,h);
    if (max_side > 1024) {
        const float scale=1024.f/max_side;
        const int nw=(int)(w*scale), nh=(int)(h*scale);
        imgio::Image ai; ai.w=w; ai.h=h; ai.rgb.resize((size_t)w*h*3);
        for (size_t i=0;i<alpha01.size();i++) ai.rgb[i*3]=ai.rgb[i*3+1]=ai.rgb[i*3+2]=alpha01[i];
        im=imgio::resize_lanczos3(im,nw,nh);
        ai=imgio::resize_lanczos3(ai,nw,nh);
        alpha01.resize((size_t)nw*nh);
        for (size_t i=0;i<alpha01.size();i++) alpha01[i]=ai.rgb[i*3];
        w=nw; h=nh;
    }
    int x0=0,y0=0,x1=w-1,y1=h-1;
    if (has_alpha) {
        x0=w; y0=h; x1=-1; y1=-1;
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) if (alpha01[(size_t)y*w+x] > .8f) {
            x0=std::min(x0,x); y0=std::min(y0,y); x1=std::max(x1,x); y1=std::max(y1,y);
        }
        if (x1 < x0 || y1 < y0) { x0=0; y0=0; x1=w-1; y1=h-1; }
        const int side=std::max(x1-x0,y1-y0);
        // Pillow Image.crop maps float coordinates through Python round() (ties-to-even), then
        // treats the right/bottom bound as exclusive.  This matters for a 1024-square matte:
        // (.5,.5,1022.5,1022.5) becomes [0,1022), not a 1023px inclusive crop.
        const double cx=.5*(x0+x1), cy=.5*(y0+y1);
        x0=(int)std::nearbyint(cx-side/2); y0=(int)std::nearbyint(cy-side/2);
        x1=(int)std::nearbyint(cx+side/2); y1=(int)std::nearbyint(cy+side/2);
    }
    const int ow=std::max(1,x1-x0), oh=std::max(1,y1-y0);
    imgio::Image cropped; cropped.w=ow; cropped.h=oh; cropped.rgb.assign((size_t)ow*oh*3,0.f);
    for (int y=0;y<oh;y++) for (int x=0;x<ow;x++) {
        const int sx=x0+x, sy=y0+y;
        if (sx<0 || sx>=w || sy<0 || sy>=h) continue;
        // Python premultiplies RGB by alpha after cropping.
        for (int ch=0;ch<3;ch++) cropped.rgb[((size_t)y*ow+x)*3+ch] =
            im.rgb[((size_t)sy*w+sx)*3+ch] * alpha01[(size_t)sy*w+sx];
    }
    stbi_image_free(p);
    return cropped;
}

int main(int argc, char** argv) {
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);
    // Direct xatlas charts with Pixal3D's chart settings are the quality default.  The old
    // precluster path remains available as `production` for a deliberately faster fallback.
    std::string model, mesh_path, image_path, out, dump_dir, unwrap="reference";
    bool condition_only=false;
    int resolution=512, texsize=2048, seed=42, decimate=0;
    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if (a=="--model" && i+1<argc) model=argv[++i];
        else if (a=="--mesh" && i+1<argc) mesh_path=argv[++i];
        else if (a=="--image" && i+1<argc) image_path=argv[++i];
        else if (a=="--out" && i+1<argc) out=argv[++i];
        else if (a=="--resolution" && i+1<argc) resolution=std::atoi(argv[++i]);
        else if (a=="--texsize" && i+1<argc) texsize=std::atoi(argv[++i]);
        else if (a=="--seed" && i+1<argc) seed=std::atoi(argv[++i]);
        else if (a=="--decimate" && i+1<argc) decimate=std::atoi(argv[++i]);
        else if (a=="--dump-dir" && i+1<argc) dump_dir=argv[++i];
        else if (a=="--unwrap" && i+1<argc) unwrap=argv[++i];
        else if (a=="--condition-only") condition_only=true;
        else { usage(); return 1; }
    }
    if (model.empty() || mesh_path.empty() || image_path.empty() || (out.empty() && !condition_only) ||
        (resolution!=512 && resolution!=1024) || (unwrap!="production" && unwrap!="reference")) { usage(); return 1; }
    setenv("PIXAL3D_GGUF_DIR", model.c_str(), 1);

    try {
        glb::Mesh mesh;
        if (!glb::read_glb(mesh_path.c_str(), mesh) || mesh.verts.empty() || mesh.faces.empty())
            throw std::runtime_error("could not read mesh GLB");
        std::printf("[native-texture] mesh %zu v / %zu f, resolution=%d, atlas=%d\\n",
                    mesh.verts.size()/3, mesh.faces.size()/3, resolution, texsize);
        preprocess_mesh(mesh.verts);

        imgio::Image source=load_texture_image(image_path);
        if (!dump_dir.empty()) { std::filesystem::create_directories(dump_dir); dump_npy(dump_dir+"/native_proc_image_chw.npy",imgio::to_chw(source),"<f4",{3,source.h,source.w}); }
        imgio::Image resized=imgio::resize_lanczos3(source, resolution, resolution);
        std::vector<float> img_chw=imgio::to_chw(resized);
        if (!dump_dir.empty()) { std::filesystem::create_directories(dump_dir); dump_npy(dump_dir+"/native_dino_input_chw.npy",img_chw,"<f4",{3,resolution,resolution}); }
        std::vector<float> img_norm=pix::imagenet_norm(img_chw, resolution);
        std::vector<float> global, patchmap;
        pix::run_dinov3(img_norm, resolution==512 ? dino::CFG512 : dino::CFG1024, true, global, patchmap);
        std::vector<float> cond=global;
        cond.insert(cond.end(), patchmap.begin(), patchmap.end());
        if (!dump_dir.empty()) { std::filesystem::create_directories(dump_dir); dump_npy(dump_dir+"/native_cond.npy",cond,"<f4",{1,(int)cond.size()/1024,1024}); }
        if (condition_only) { std::printf("[native-texture] condition-only complete\n"); return 0; }
        std::printf("[native-texture] DINOv3: %zu conditioning tokens\\n", cond.size()/1024);

        std::vector<int32_t> faces(mesh.faces.size());
        for (size_t i=0;i<faces.size();i++) faces[i]=(int32_t)mesh.faces[i];
        int gs[3]={resolution,resolution,resolution};
        const float amin[3]={-.5f,-.5f,-.5f}, amax[3]={.5f,.5f,.5f};
        vox::VoxelOut voxels=vox::mesh_to_flexible_dual_grid(mesh.verts.data(), (int)mesh.verts.size()/3,
            faces.data(), (int)faces.size()/3, gs, amin, amax, 1.f, .2f, 1e-2f);
        if (!voxels.N) throw std::runtime_error("voxelizer returned no surface cells");
        std::vector<int32_t> coords((size_t)voxels.N*4);
        std::vector<float> feats6((size_t)voxels.N*6);
        for (int i=0;i<voxels.N;i++) for (int c=0;c<3;c++) {
            coords[(size_t)i*4+c+1]=voxels.coords[(size_t)i*3+c];
            feats6[(size_t)i*6+c]=voxels.dual[(size_t)i*3+c]-.5f;
            feats6[(size_t)i*6+c+3]=(float)voxels.intersected[(size_t)i*3+c]-.5f;
        }
        if (!dump_dir.empty()) { dump_npy(dump_dir+"/native_voxel_coords.npy",coords,"<i4",{voxels.N,4}); dump_npy(dump_dir+"/native_voxel_feats6.npy",feats6,"<f4",{voxels.N,6}); }
        std::vector<int32_t> slat_coords;
        std::vector<float> slat=senc::shape_slat_encode(coords, feats6, "weights_npy/shape_enc", true, &slat_coords);
        const int M=(int)slat_coords.size()/4;
        if (!M) throw std::runtime_error("shape encoder returned no SLat tokens");
        if (!dump_dir.empty()) { dump_npy(dump_dir+"/native_shape_slat_coords.npy",slat_coords,"<i4",{M,4}); dump_npy(dump_dir+"/native_shape_slat_feats.npy",slat,"<f4",{M,32}); }
        std::printf("[native-texture] flexible grid %d -> shape SLat %d tokens\\n", voxels.N, M);

        std::vector<float> shape_mean=load_norm(model,"shape_slat_norm_mean"), shape_std=load_norm(model,"shape_slat_norm_std");
        std::vector<float> tex_mean=load_norm(model,"tex_slat_norm_mean"), tex_std=load_norm(model,"tex_slat_norm_std");
        for (size_t i=0;i<slat.size();i++) slat[i]=(slat[i]-shape_mean[i%32])/shape_std[i%32];
        std::vector<int32_t> cxyz((size_t)M*3);
        for (int i=0;i<M;i++) for (int c=0;c<3;c++) cxyz[(size_t)i*3+c]=slat_coords[(size_t)i*4+c+1];
        std::vector<float> tex;
        {
            trandn::Generator gen((uint64_t)seed);
            std::vector<float> noise=gen.randn((int64_t)M*32), x64((size_t)M*64);
            // The two lattice sizes use separately trained texture-flow weights.  Selecting 1024 here
            // for a 512 comparison quietly invalidates parity even if every operation is native.
            const char* flow_weights = resolution==512 ? "weights_npy/trellis2_tex_512" : "weights_npy/trellis2_tex_1024";
            M1Harness H(flow_weights, 4096, true);
            ggml_context* ctx=H.ctx;
            int64_t xn[4]={64,M,1,1}, tn[4]={1,1,1,1}, cn[4]={1024,(int64_t)cond.size()/1024,1,1};
            ggml_tensor* xin=H.input("x",2,xn), *tin=H.input("t",1,tn), *cin=H.input("cond",2,cn);
            ggml_tensor* pred=texdit::build_tex_dit_cross_forward(ctx,H,M,(int)cn[1],xin,tin,cin,cxyz.data());
            ggml_set_output(pred); ggml_cgraph* graph=new_graph(ctx,65536); ggml_build_forward_expand(graph,pred);
            H.alloc_and_upload(graph); H.upload_input_raw(cin,cond);
            auto forward=[&](const std::vector<float>& x, float t, bool) {
                for (int i=0;i<M;i++) for (int c=0;c<32;c++) { x64[(size_t)i*64+c]=x[(size_t)i*32+c]; x64[(size_t)i*64+c+32]=slat[(size_t)i*32+c]; }
                H.upload_input_raw(xin,x64); H.upload_input_raw(tin,std::vector<float>{t}); H.compute(graph);
                std::vector<float> r((size_t)M*32); ggml_backend_tensor_get(pred,r.data(),0,r.size()*sizeof(float)); return r;
            };
            tex=geo::flow_sampler((int64_t)M*32,noise,1e-5f,1.f,0.f,3.f,.6,.9,12,forward,"native tex");
        } // Release the large texture-flow graph and its GPU weights before M6 PBR decode.
        for (size_t i=0;i<tex.size();i++) tex[i]=tex[i]*tex_std[i%32]+tex_mean[i%32];
        if (!dump_dir.empty()) dump_npy(dump_dir+"/native_tex_slat_feats.npy",tex,"<f4",{M,32});
        std::vector<std::vector<uint8_t>> subs=senc::build_guide_subs(coords,slat_coords);
        std::vector<int32_t> pbr_coords;
        std::vector<float> pbr=svpg::m6_tex_decode(slat_coords,tex,subs,"weights_npy/tex_dec",&pbr_coords);
        // A CUDA allocation failure used to be swallowed by the decoder and surfaced as a
        // perfectly neutral (0.5, 0.5, 0.5) atlas.  That is never a useful generated texture;
        // fail closed so a high-resolution run cannot silently replace a good production asset.
        if (pbr.empty()) throw std::runtime_error("PBR decoder returned no voxels");
        float rgb_min=1e30f, rgb_max=-1e30f;
        for (size_t i=0;i<pbr.size();i+=6) for (int c=0;c<3;c++) {
            rgb_min=std::min(rgb_min,pbr[i+c]); rgb_max=std::max(rgb_max,pbr[i+c]);
        }
        if (rgb_max-rgb_min < 1e-4f)
            throw std::runtime_error("PBR decoder collapsed to a neutral field (likely CUDA memory pressure)");
        if (const char* f=std::getenv("TEX_PBR_OUTLIER")) {
            float threshold=(float)atof(f);
            size_t changed=texatlas::filter_pbr_rgb_outliers(pbr,pbr_coords,threshold);
            std::printf("[native-texture] PBR RGB outlier cleanup threshold=%.3f: %zu / %zu voxels\n",
                        threshold,changed,pbr.size()/6);
        }
        if (!dump_dir.empty()) { dump_npy(dump_dir+"/native_pbr_coords.npy",pbr_coords,"<i4",{(int)pbr_coords.size()/4,4}); dump_npy(dump_dir+"/native_pbr_feats.npy",pbr,"<f4",{(int)pbr.size()/6,6}); }
        std::printf("[native-texture] PBR volume: %zu voxels\\n", pbr.size()/6);

        // The production unwrap is deliberately fast and chart-safe for every model.  The
        // reference mode removes that pre-cluster and uses Pixal3D/CuMesh xatlas chart settings,
        // giving a like-for-like atlas-quality A/B while keeping all inference native.
        const bool reference_unwrap = unwrap=="reference";
        if (reference_unwrap) {
            setenv("ATL_PYREF_XATLAS", "1", 1);
            std::printf("[native-texture] reference unwrap: direct mesh xatlas, Pixal3D chart settings\\n");
        }
        texatlas::BakedTexture baked=texatlas::bake(mesh.verts,mesh.faces,pbr,pbr_coords,resolution,texsize,decimate,
                                                    reference_unwrap ? 0 : 4,true,8,!reference_unwrap);
        restore_mesh_frame(baked.verts); restore_mesh_frame(baked.normals);
        if (!glb::write_glb_textured(out.c_str(),baked.verts,baked.normals,baked.uvs,baked.faces,
                                     baked.base_color,baked.metal_rough,baked.tw,baked.th))
            throw std::runtime_error("could not write output GLB");
        std::printf("[native-texture] DONE: %s (%d charts, %dx%d atlas)\\n",out.c_str(),baked.chart_count,baked.tw,baked.th);
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL: %s\\n",e.what()); return 1; }
    return 0;
}
