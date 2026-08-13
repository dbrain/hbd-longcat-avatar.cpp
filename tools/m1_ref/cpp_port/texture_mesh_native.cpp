// texture_mesh_native -- native TRELLIS-2 texture-only pass for an existing mesh.
//
// This is the production counterpart of texture_mesh.py.  It deliberately starts with the
// FINAL (e.g. UltraShape-refined) mesh instead of rebaking the geometry pass's low-frequency PBR
// volume.  The chain is:
//   mesh -> Trellis2 canonical frame -> flexible dual grid -> shape SLat encoder
//        -> DINOv3 full-token cross conditioning -> Trellis2 texture SLat DiT
//        -> PBR decoder -> UV bake.
// The projection-conditioned Pixal texture checkpoint remains available only as an explicit
// diagnostic mode; it is not the generic arbitrary-mesh texturing contract.
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
#include <fstream>
#include <string>
#include <vector>

static void usage() {
    std::printf("usage: texture_mesh_native --model <geo_gguf_dir> --mesh <in.glb> --image <matte.png> --out <out.glb>\n"
                "                           [--resolution 1024] [--texsize N] [--seed N] [--noise-npy reference_noise.npy] [--texture-model generic-cross|pixal-proj] [--encoder-decimate F] [--decimate F] [--dump-dir DIR]\n"
                "                           [--camera camera_provenance.txt | --cam radians --dist units --scale value] [--voxelizer-only|--shape-encoder-only] [--shape-encoder-input-dir DIR]\n"
                "                           [--unwrap production|reference] [--atlas-out base_color.png] [--status-file PATH]\n");
}

static float camera_value(const std::string& path, const char* key, float fallback) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read camera provenance: " + path);
    const std::string prefix=std::string(key)+"=";
    std::string line;
    while (std::getline(f,line)) if (line.rfind(prefix,0)==0) return std::stof(line.substr(prefix.size()));
    throw std::runtime_error("camera provenance lacks " + std::string(key) + ": " + path);
}

// The runner may be interrupted after this child has completed (for example a detached terminal
// session).  Record the artifact outcome here, at the point that the GLB and atlas have actually
// been written, so the final truth does not depend on the parent shell's exit status.
static void append_artifact_status(const std::string& path, const char* state, int code) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::app);
    if (!f) return;
    f << "artifact_state=" << state << '\n';
    f << "artifact_exit_code=" << code << '\n';
}

static void native_stage(const char* name) {
    const char* path=std::getenv("TEX_STAGE_LOG");
    if (!path || !*path) return;
    std::ofstream f(path,std::ios::app);
    if (f) f << "stage=" << name << '\n';
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
    // trimesh exposes GLB positions as float64, and the official pipeline performs this
    // normalization in NumPy float64 before torch casts the finished vertices to float32.
    // Keeping this arithmetic in float used to move boundary-aligned triangles by one ULP,
    // creating a handful of different o_voxel cells despite the same source GLB.
    double mn[3] = {1e300,1e300,1e300}, mx[3] = {-1e300,-1e300,-1e300};
    for (size_t i=0;i<v.size();i+=3) for (int c=0;c<3;c++) {
        const double value = v[i+c];
        mn[c] = std::min(mn[c], value); mx[c] = std::max(mx[c], value);
    }
    double center[3], extent=0.;
    for (int c=0;c<3;c++) { center[c]=0.5f*(mn[c]+mx[c]); extent=std::max(extent,mx[c]-mn[c]); }
    const double scale = 0.99999 / std::max(extent, 1e-12);
    for (size_t i=0;i<v.size();i+=3) {
        const double x=(v[i]-center[0])*scale, y=(v[i+1]-center[1])*scale, z=(v[i+2]-center[2])*scale;
        v[i]=(float)x; v[i+1]=(float)-z; v[i+2]=(float)y;
    }
}

static void restore_mesh_frame(std::vector<float>& v) {
    for (size_t i=0;i<v.size();i+=3) { float y=v[i+1], z=v[i+2]; v[i+1]=z; v[i+2]=-y; }
}

// Trellis2TexturingPipeline receives a PIL image, optionally crops its transparent subject, then
// DinoV3FeatureExtractor does the final square Lanczos resize.  The runbook inputs are already
// matted; this keeps their transparent padding from changing the image conditioning.
static bool env_enabled(const char* key, bool default_value) {
    const char* value=std::getenv(key);
    if (!value || !*value) return default_value;
    return std::strcmp(value,"0") && std::strcmp(value,"false") && std::strcmp(value,"False");
}

static imgio::Image load_texture_image(const std::string& path, std::string* input_contract=nullptr) {
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
    // The runbook's canonical geometry frame is often an opaque RGB black matte.  Its black
    // background is *not* an alpha channel: synthesising alpha from RGB brightness erased black
    // hair, pupils, eyeliner, boots and dark clothing before DINO saw them.  Preserve the exact
    // canonical frame by default; the geometry and texture conditioners then see the same image.
    // Actual RGBA sources still use their true alpha/crop contract.  The old luminance-mask path
    // remains an explicit diagnostic compatibility A/B only.
    bool black_matte=false;
    if (!has_alpha) {
        int border=0, black=0;
        for (int y=0;y<h;y++) for (int x=0;x<w;x++) if (x==0||y==0||x==w-1||y==h-1) {
            border++; const float* q=&im.rgb[((size_t)y*w+x)*3];
            if (std::max({q[0],q[1],q[2]}) < .035f) black++;
        }
        black_matte=border && black > border*95/100;
        if (black_matte && !env_enabled("TEX_BLACK_MATTE_PRESERVE",true)) {
            for (size_t i=0;i<alpha01.size();i++) {
                const float* q=&im.rgb[i*3]; float m=std::max({q[0],q[1],q[2]});
                alpha01[i]=std::max(0.f,std::min(1.f,(m-.01f)/.06f));
            }
            has_alpha=true;
            if (input_contract) *input_contract="opaque-black-matte-luminance-alpha-compat";
        } else if (black_matte) {
            if (input_contract) *input_contract="opaque-black-matte-preserved";
            std::printf("[native-texture] input: preserve opaque black matte (no luminance-derived alpha)\n");
        } else if (input_contract) {
            *input_contract="opaque-image-full-frame";
        }
    } else if (input_contract) {
        *input_contract="rgba-alpha-crop";
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
    // Pillow crops use exclusive right/bottom coordinates.  Preserve the entire opaque input
    // verbatim: using w-1/h-1 here silently discarded the final row and column, which then
    // changed every DINO feature during the square resize.
    int x0=0,y0=0,x1=w,y1=h;
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
    std::string model, mesh_path, image_path, out, dump_dir, atlas_out, status_file, camera_file, noise_npy, shape_encoder_input_dir, unwrap="reference", texture_model="generic-cross";
    bool condition_only=false, voxelizer_only=false, shape_encoder_only=false;
    int resolution=1024, texsize=2048, seed=42, encoder_decimate=0, decimate=0;
    float cam=.7332379387484828f, dist=1.3021559715270996f, mesh_scale=1.f;
    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if (a=="--model" && i+1<argc) model=argv[++i];
        else if (a=="--mesh" && i+1<argc) mesh_path=argv[++i];
        else if (a=="--image" && i+1<argc) image_path=argv[++i];
        else if (a=="--out" && i+1<argc) out=argv[++i];
        else if (a=="--resolution" && i+1<argc) resolution=std::atoi(argv[++i]);
        else if (a=="--texsize" && i+1<argc) texsize=std::atoi(argv[++i]);
        else if (a=="--seed" && i+1<argc) seed=std::atoi(argv[++i]);
        else if (a=="--noise-npy" && i+1<argc) noise_npy=argv[++i];
        else if (a=="--texture-model" && i+1<argc) texture_model=argv[++i];
        else if (a=="--encoder-decimate" && i+1<argc) encoder_decimate=std::atoi(argv[++i]);
        else if (a=="--decimate" && i+1<argc) decimate=std::atoi(argv[++i]);
        else if (a=="--dump-dir" && i+1<argc) dump_dir=argv[++i];
        else if (a=="--unwrap" && i+1<argc) unwrap=argv[++i];
        else if (a=="--atlas-out" && i+1<argc) atlas_out=argv[++i];
        else if (a=="--status-file" && i+1<argc) status_file=argv[++i];
        else if (a=="--camera" && i+1<argc) camera_file=argv[++i];
        else if (a=="--shape-encoder-input-dir" && i+1<argc) shape_encoder_input_dir=argv[++i];
        else if (a=="--cam" && i+1<argc) cam=std::stof(argv[++i]);
        else if (a=="--dist" && i+1<argc) dist=std::stof(argv[++i]);
        else if (a=="--scale" && i+1<argc) mesh_scale=std::stof(argv[++i]);
        else if (a=="--condition-only") condition_only=true;
        else if (a=="--voxelizer-only") voxelizer_only=true;
        else if (a=="--shape-encoder-only") shape_encoder_only=true;
        else { usage(); return 1; }
    }
    if (model.empty() || mesh_path.empty() || image_path.empty() || (out.empty() && !condition_only) ||
        resolution!=1024 || (unwrap!="production" && unwrap!="reference") ||
        (texture_model!="generic-cross" && texture_model!="pixal-proj")) { usage(); return 1; }
    if (!camera_file.empty()) {
        cam=camera_value(camera_file,"cam_angle_x_rad",cam);
        dist=camera_value(camera_file,"camera_distance",dist);
        mesh_scale=camera_value(camera_file,"mesh_scale",mesh_scale);
    }
    if (!(cam>.05f && cam<3.f && dist>0.f && mesh_scale>0.f)) {
        std::fprintf(stderr,"FAIL: invalid camera contract (cam=%g dist=%g scale=%g)\n",cam,dist,mesh_scale); return 1;
    }
    if (!condition_only && atlas_out.empty()) atlas_out=out+".atlas.png";
    if (!condition_only && !std::getenv("TEX_STAGE_LOG")) {
        const std::string stage_log=out+".stage-log.txt";
        std::ofstream clear(stage_log,std::ios::trunc);
        if (!clear) { std::fprintf(stderr,"FAIL: cannot create stage log: %s\n",stage_log.c_str()); return 1; }
        setenv("TEX_STAGE_LOG",stage_log.c_str(),1);
    }
    setenv("PIXAL3D_GGUF_DIR", model.c_str(), 1);

    try {
        // `generic-cross` is the exact model contract of Trellis2TexturingPipeline: full DINO
        // image-token cross-attention, the generic shape encoder, and the matching M6 decoder.
        // Keep Pixal's projection model as an explicit diagnostic only; silently mixing these
        // two checkpoints made prior native/Python texture comparisons meaningless.
        if (!condition_only && texture_model=="generic-cross" &&
            !std::filesystem::is_directory("weights_npy/trellis2_tex_1024"))
            throw std::runtime_error("missing generic Trellis2 cross checkpoint: weights_npy/trellis2_tex_1024");
        if (!condition_only && texture_model=="pixal-proj") {
            const std::string tex_checkpoint=model+"/slat_flow_imgshape2tex_1024.gguf";
            if (!std::filesystem::is_regular_file(tex_checkpoint))
                throw std::runtime_error("missing required Pixal projection texture checkpoint: "+tex_checkpoint);
        }
        native_stage("start");
        glb::Mesh mesh;
        if (!glb::read_glb(mesh_path.c_str(), mesh) || mesh.verts.empty() || mesh.faces.empty())
            throw std::runtime_error("could not read mesh GLB");
        // Delivery meshes contain UV seam duplicates and can be much denser than the sparse
        // encoder's trained surface-token regime.  Cap the encoder input before voxelisation;
        // the existing --decimate remains the independent final atlas/LOD setting.
        if (encoder_decimate > 0 && (int)mesh.faces.size()/3 > encoder_decimate) {
            std::vector<float> dverts;
            std::vector<int64_t> dfaces;
            texatlas::decimate(mesh.verts, mesh.faces, (size_t)encoder_decimate, dverts, dfaces, true);
            mesh.verts.swap(dverts);
            mesh.faces.swap(dfaces);
            std::printf("[native-texture] encoder mesh decimated to %zu v / %zu f\\n",
                        mesh.verts.size()/3, mesh.faces.size()/3);
        }
        std::printf("[native-texture] mesh %zu v / %zu f, resolution=%d, atlas=%d\\n",
                    mesh.verts.size()/3, mesh.faces.size()/3, resolution, texsize);
        preprocess_mesh(mesh.verts);
        native_stage("mesh_preprocessed");

        std::string texture_input_contract;
        imgio::Image source=load_texture_image(image_path,&texture_input_contract);
        if (!dump_dir.empty()) {
            std::filesystem::create_directories(dump_dir);
            dump_npy(dump_dir+"/native_preprocessed_vertices.npy",mesh.verts,"<f4",{(int)mesh.verts.size()/3,3});
            dump_npy(dump_dir+"/native_preprocessed_faces.npy",mesh.faces,"<i8",{(int)mesh.faces.size()/3,3});
            dump_npy(dump_dir+"/native_proc_image_chw.npy",imgio::to_chw(source),"<f4",{3,source.h,source.w});
            std::ofstream input_meta(dump_dir+"/native_input_contract.txt",std::ios::trunc);
            if (!input_meta) throw std::runtime_error("could not write input contract metadata");
            input_meta << texture_input_contract << '\n';
        }
        std::printf("[native-texture] texture input contract: %s\n",texture_input_contract.c_str());
        imgio::Image resized=imgio::resize_lanczos3(source, resolution, resolution);
        std::vector<float> img_chw=imgio::to_chw(resized);
        if (!dump_dir.empty()) { std::filesystem::create_directories(dump_dir); dump_npy(dump_dir+"/native_dino_input_chw.npy",img_chw,"<f4",{3,resolution,resolution}); }
        std::vector<float> img_norm=pix::imagenet_norm(img_chw, resolution);
        std::vector<float> global, patchmap;
        pix::run_dinov3(img_norm, dino::CFG1024, true, global, patchmap);
        native_stage("dino_complete");
        if (!dump_dir.empty()) { std::filesystem::create_directories(dump_dir); dump_npy(dump_dir+"/native_tex_global.npy",global,"<f4",{1,5,1024}); }
        // Persist the exact generic flow input before the condition-only exit as well.  This keeps
        // the small DINO parity gate useful without having to run the encoder or texture flow.
        std::vector<float> dino_cross=global;
        dino_cross.insert(dino_cross.end(),patchmap.begin(),patchmap.end());
        if (!dump_dir.empty()) dump_npy(dump_dir+"/native_tex_cross_cond.npy",dino_cross,"<f4",
                                        {1,5+(int)(patchmap.size()/dino::HID),dino::HID});
        if (condition_only) {
            append_artifact_status(status_file, "succeeded", 0);
            std::printf("[native-texture] condition-only complete\n"); return 0;
        }
        if (texture_model=="pixal-proj") {
            std::printf("[native-texture] Pixal projection texture: FOV=%.4f rad dist=%.6f scale=%.6f%s\\n",
                        cam,dist,mesh_scale,camera_file.empty()?" (default contract)":" (provenance)");
        } else {
            std::printf("[native-texture] generic Trellis2 cross texture: %d DINO tokens (no camera projection)\\n",
                        5+(int)(patchmap.size()/dino::HID));
        }

        std::vector<int32_t> coords;
        std::vector<float> feats6;
        int voxel_N=0;
        if (!shape_encoder_input_dir.empty()) {
            NpyArray c=npy_load(shape_encoder_input_dir+"/python_voxel_coords.npy");
            NpyArray f=npy_load(shape_encoder_input_dir+"/python_voxel_feats6.npy");
            if (c.descr!="<i4" || f.descr!="<f4" || c.shape.size()!=2 || f.shape.size()!=2 ||
                c.shape[1]!=4 || f.shape[1]!=6 || c.shape[0]!=f.shape[0])
                throw std::runtime_error("--shape-encoder-input-dir requires official [N,4] int32 coords and [N,6] float32 feats");
            coords.assign(c.i32(),c.i32()+c.numel()); feats6.assign(f.f32(),f.f32()+f.numel());
            voxel_N=(int)c.shape[0];
            std::printf("[native-texture] shape encoder uses supplied official voxel boundary: %s\\n",shape_encoder_input_dir.c_str());
        } else {
            std::vector<int32_t> faces(mesh.faces.size());
            for (size_t i=0;i<faces.size();i++) faces[i]=(int32_t)mesh.faces[i];
            int gs[3]={resolution,resolution,resolution};
            const float amin[3]={-.5f,-.5f,-.5f}, amax[3]={.5f,.5f,.5f};
            vox::VoxelOut voxels=vox::mesh_to_flexible_dual_grid(mesh.verts.data(), (int)mesh.verts.size()/3,
                faces.data(), (int)faces.size()/3, gs, amin, amax, 1.f, .2f, 1e-2f);
            if (!voxels.N) throw std::runtime_error("voxelizer returned no surface cells");
            voxel_N=voxels.N; coords.resize((size_t)voxel_N*4); feats6.resize((size_t)voxel_N*6);
            for (int i=0;i<voxel_N;i++) for (int c=0;c<3;c++) {
                coords[(size_t)i*4+c+1]=voxels.coords[(size_t)i*3+c];
                // FlexiDualGridVaeEncoder applies this centring itself before its sparse backbone.
                // This C++ port feeds that backbone directly, so preserve the upstream centred contract
                // here rather than passing the raw [0,1] o_voxel values a second time.
                feats6[(size_t)i*6+c]=voxels.dual[(size_t)i*3+c]-.5f;
                feats6[(size_t)i*6+c+3]=(float)voxels.intersected[(size_t)i*3+c]-.5f;
            }
        }
        native_stage("voxel_complete");
        if (!voxel_N) throw std::runtime_error("voxelizer returned no surface cells");
        if (!dump_dir.empty()) { dump_npy(dump_dir+"/native_voxel_coords.npy",coords,"<i4",{voxel_N,4}); dump_npy(dump_dir+"/native_voxel_feats6.npy",feats6,"<f4",{voxel_N,6}); }
        if (voxelizer_only) {
            append_artifact_status(status_file, "succeeded", 0);
            std::printf("[native-texture] voxelizer-only complete\\n");
            return 0;
        }
        std::vector<int32_t> slat_coords;
        std::vector<float> slat=senc::shape_slat_encode(coords, feats6, "weights_npy/shape_enc", true, &slat_coords);
        native_stage("shape_slat_complete");
        const int M=(int)slat_coords.size()/4;
        if (!M) throw std::runtime_error("shape encoder returned no SLat tokens");
        if (!dump_dir.empty()) { dump_npy(dump_dir+"/native_shape_slat_coords.npy",slat_coords,"<i4",{M,4}); dump_npy(dump_dir+"/native_shape_slat_feats.npy",slat,"<f4",{M,32}); }
        std::printf("[native-texture] flexible grid %d -> shape SLat %d tokens\\n", voxel_N, M);
        if (shape_encoder_only) {
            append_artifact_status(status_file, "succeeded", 0);
            std::printf("[native-texture] shape-encoder-only complete\\n");
            return 0;
        }

        std::vector<float> shape_mean=load_norm(model,"shape_slat_norm_mean"), shape_std=load_norm(model,"shape_slat_norm_std");
        std::vector<float> tex_mean=load_norm(model,"tex_slat_norm_mean"), tex_std=load_norm(model,"tex_slat_norm_std");
        for (size_t i=0;i<slat.size();i++) slat[i]=(slat[i]-shape_mean[i%32])/shape_std[i%32];
        std::vector<int32_t> cxyz((size_t)M*3);
        for (int i=0;i<M;i++) for (int c=0;c<3;c++) cxyz[(size_t)i*3+c]=slat_coords[(size_t)i*4+c+1];
        std::vector<float> tex;
        if (texture_model=="pixal-proj") {
            std::vector<float> naf_hr=pix::run_naf(img_chw,patchmap,naf::CFG1024,true);
            native_stage("naf_complete");
            geo::ProjCam texture_cam(cam,dist,mesh_scale);
            std::vector<float> proj_cond=geo::proj_cond_shape(slat_coords.data(),M,patchmap.data(),64,64,
                naf_hr.data(),512,512,dino::HID,resolution/16,1024,texture_cam);
            native_stage("projection_condition_complete");
            if (!dump_dir.empty()) dump_npy(dump_dir+"/native_tex_proj_cond.npy",proj_cond,"<f4",{M,slatdit::PROJ_IN});
            std::vector<float> noise;
            if (!noise_npy.empty()) {
                NpyArray a=npy_load(noise_npy);
                if (a.descr!="<f4" || a.shape.size()!=2 || a.shape[0]!=M || a.shape[1]!=32)
                    throw std::runtime_error("--noise-npy must be a float32 [shape_slat_tokens,32] tensor");
                noise.assign(a.f32(),a.f32()+a.numel());
                std::printf("[native-texture] using supplied diagnostic flow noise: %s\n",noise_npy.c_str());
            } else {
                trandn::Generator gen((uint64_t)seed);
                noise=gen.randn((int64_t)M*32);
            }
            std::vector<float> x64((size_t)M*64);
            M1Harness H(pix::TEXFLOW_PROJ_W, 2048, true);
            ggml_context* ctx=H.ctx;
            int64_t xn[4]={64,M,1,1}, tn[4]={1,1,1,1}, gn[4]={1024,5,1,1}, pn[4]={slatdit::PROJ_IN,M,1,1};
            ggml_tensor* xin=H.input("x",2,xn), *tin=H.input("t",1,tn), *gin=H.input("global",2,gn), *pin=H.input("proj",2,pn);
            ggml_tensor* pred=slatdit::build_slat_dit_forward(ctx,H,M,xin,tin,gin,pin,cxyz.data());
            ggml_set_output(pred); ggml_cgraph* graph=new_graph(ctx,65536); ggml_build_forward_expand(graph,pred);
            H.alloc_and_upload(graph); H.upload_input_raw(gin,global); H.upload_input_raw(pin,proj_cond);
            auto forward=[&](const std::vector<float>& x, float t, bool) {
                for (int i=0;i<M;i++) for (int c=0;c<32;c++) { x64[(size_t)i*64+c]=x[(size_t)i*32+c]; x64[(size_t)i*64+c+32]=slat[(size_t)i*32+c]; }
                H.upload_input_raw(xin,x64); H.upload_input_raw(tin,std::vector<float>{t}); H.compute(graph);
                std::vector<float> r((size_t)M*32); ggml_backend_tensor_get(pred,r.data(),0,r.size()*sizeof(float)); return r;
            };
            tex=geo::flow_sampler((int64_t)M*32,noise,1e-5f,1.f,0.f,3.f,.6,.9,12,forward,"native Pixal proj tex");
        } else {
            // Exact Trellis2TexturingPipeline contract: five DINO global tokens followed by every
            // image patch token, fed through the generic cross-attention texture flow.  This is the
            // reference model, unlike Pixal's image-to-3D projection-conditioned texture flow above.
            const int Ntok=5+(int)(patchmap.size()/dino::HID);
            const std::vector<float>& cond=dino_cross;
            native_stage("cross_condition_complete");
            trandn::Generator gen((uint64_t)seed);
            std::vector<float> noise=gen.randn((int64_t)M*32), x64((size_t)M*64);
            M1Harness H(pix::TEXFLOW_W,4096,true);
            ggml_context* ctx=H.ctx;
            int64_t xn[4]={64,M,1,1}, tn[4]={1,1,1,1}, cn[4]={dino::HID,Ntok,1,1};
            ggml_tensor* xin=H.input("x",2,xn), *tin=H.input("t",1,tn), *cin=H.input("cond",2,cn);
            ggml_tensor* pred=texdit::build_tex_dit_cross_forward(ctx,H,M,Ntok,xin,tin,cin,cxyz.data());
            ggml_set_output(pred); ggml_cgraph* graph=new_graph(ctx,65536); ggml_build_forward_expand(graph,pred);
            H.alloc_and_upload(graph); H.upload_input_raw(cin,cond);
            auto forward=[&](const std::vector<float>& x,float t,bool) {
                for (int i=0;i<M;i++) for (int c=0;c<32;c++) { x64[(size_t)i*64+c]=x[(size_t)i*32+c]; x64[(size_t)i*64+c+32]=slat[(size_t)i*32+c]; }
                H.upload_input_raw(xin,x64); H.upload_input_raw(tin,std::vector<float>{t}); H.compute(graph);
                std::vector<float> r((size_t)M*32); ggml_backend_tensor_get(pred,r.data(),0,r.size()*sizeof(float)); return r;
            };
            tex=geo::flow_sampler((int64_t)M*32,noise,1e-5f,1.f,0.f,3.f,.6,.9,12,forward,"native Trellis2 cross tex");
        } // Release the large texture-flow graph and its GPU weights before M6 PBR decode.
        native_stage("texture_flow_complete");
        for (size_t i=0;i<tex.size();i++) tex[i]=tex[i]*tex_std[i%32]+tex_mean[i%32];
        if (!dump_dir.empty()) dump_npy(dump_dir+"/native_tex_slat_feats.npy",tex,"<f4",{M,32});
        std::vector<std::vector<uint8_t>> subs=senc::build_guide_subs(coords,slat_coords);
        std::vector<int32_t> pbr_coords;
        std::vector<float> pbr=svpg::m6_tex_decode(slat_coords,tex,subs,"weights_npy/tex_dec",&pbr_coords);
        native_stage("pbr_decode_complete");
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
        // Direct reference charts preserve the model's small details, so remove only genuinely
        // isolated RGB voxels before they become atlas freckles.  This is a 3-D neighbourhood
        // operation (not UV blur), and an explicit TEX_PBR_OUTLIER=0 remains a strict A/B opt-out.
        const bool reference_unwrap = unwrap=="reference";
        if (reference_unwrap) setenv("TEX_PBR_OUTLIER", "0.10", 0);
        float pbr_outlier_threshold=0.f;
        if (const char* f=std::getenv("TEX_PBR_OUTLIER")) {
            pbr_outlier_threshold=(float)atof(f);
            size_t changed=texatlas::filter_pbr_rgb_outliers(pbr,pbr_coords,pbr_outlier_threshold);
            std::printf("[native-texture] PBR RGB outlier cleanup threshold=%.3f: %zu / %zu voxels\n",
                        pbr_outlier_threshold,changed,pbr.size()/6);
        }
        if (!dump_dir.empty()) {
            dump_npy(dump_dir+"/native_pbr_coords.npy",pbr_coords,"<i4",{(int)pbr_coords.size()/4,4});
            dump_npy(dump_dir+"/native_pbr_feats.npy",pbr,"<f4",{(int)pbr.size()/6,6});
            // LOD rebakes consume this already-cleaned PBR volume.  Persist the exact cleanup
            // threshold so they do not accidentally apply a non-idempotent 3-D median twice.
            std::ofstream cleanup_meta(dump_dir+"/native_pbr_outlier_threshold.txt", std::ios::trunc);
            if (!cleanup_meta) throw std::runtime_error("could not write native PBR cleanup metadata");
            cleanup_meta << pbr_outlier_threshold << '\n';
        }
        std::printf("[native-texture] PBR volume: %zu voxels\\n", pbr.size()/6);

        // The production unwrap is deliberately fast and chart-safe for every model.  The
        // reference mode removes that pre-cluster and uses Pixal3D/CuMesh xatlas chart settings,
        // giving a like-for-like atlas-quality A/B while keeping all inference native.
        if (reference_unwrap) {
            // These are the reference-quality bake defaults established by the native parity
            // harness. The old defaults protected pre-clustered, tiny-island atlases; direct
            // xatlas charts instead need anti-aliased coverage, chart-topology normals, and
            // complete gutter repair. An explicit environment A/B setting still wins.
            auto bake_default=[](const char* key, const char* value) { setenv(key, value, 0); };
            setenv("ATL_PYREF_XATLAS", "1", 1);
            bake_default("TEX_RASTER_SS", "2");
            bake_default("TEX_TOPO_NORMALS", "1");
            bake_default("TEX_TELEA_INPAINT", "1");
            bake_default("TEX_TELEA_RADIUS", "4");
            bake_default("TEX_INPAINT_ITERS", "16");
            bake_default("TEX_FILL_BACKGROUND", "1");
            std::printf("[native-texture] reference unwrap: direct mesh xatlas, Pixal3D chart settings\\n");
        }
        texatlas::BakedTexture baked=texatlas::bake(mesh.verts,mesh.faces,pbr,pbr_coords,resolution,texsize,decimate,
                                                    reference_unwrap ? 0 : 4,true,8,!reference_unwrap);
        native_stage("atlas_complete");
        restore_mesh_frame(baked.verts); restore_mesh_frame(baked.normals);
        if (!glb::write_glb_textured(out.c_str(),baked.verts,baked.normals,baked.uvs,baked.faces,
                                     baked.base_color,baked.metal_rough,baked.tw,baked.th))
            throw std::runtime_error("could not write output GLB");
        if (!stbi_write_png(atlas_out.c_str(), baked.tw, baked.th, 4, baked.base_color.data(), baked.tw*4))
            throw std::runtime_error("could not write baseColor atlas: "+atlas_out);
        if (!texatlas::write_quality_report(out+".texture-qc.txt", baked))
            throw std::runtime_error("could not write texture quality report: "+out+".texture-qc.txt");
        native_stage("write_complete");
        append_artifact_status(status_file, "succeeded", 0);
        std::printf("[native-texture] DONE: %s (%d charts, %dx%d atlas)\\n",out.c_str(),baked.chart_count,baked.tw,baked.th);
        std::printf("[native-texture] baseColor atlas: %s\\n",atlas_out.c_str());
        std::printf("[native-texture] texture QC: %s.texture-qc.txt\\n",out.c_str());
    } catch (const std::exception& e) {
        append_artifact_status(status_file, "failed", 1);
        std::fprintf(stderr,"FAIL: %s\\n",e.what()); return 1;
    }
    return 0;
}
