// texture_rebake_native -- CPU-only native re-atlas of an already decoded PBR volume.
//
// Lets the runbook A/B chart-safe cleanup, UV settings, and atlas size without repeating DINO,
// texture diffusion, or M6 decode.  Input dumps are emitted by texture_mesh_native --dump-dir.
// No Python is used at runtime.
#include "tex_atlas.hpp"
#include "glb_reader.hpp"
#include "glb_textured.hpp"
#include "../../sparse_spike/npy.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void usage() {
    std::printf("usage: texture_rebake_native --mesh refined.glb (--pbr-dir native-dump | --pbr-cache image_to_rig_stage) --out out.glb\\n"
                "                             [--resolution N (PBR lattice, multiple of 16; default = the cache's own lattice)]\\n"
                "                             [--texsize N] [--decimate faces]\\n"
                "                             [--unwrap production|reference] [--atlas-out base_color.png]\\n"
                "                             [--bake projection|volume-trilinear (cache path; default projection)]\\n");
}

static std::vector<uint8_t> load_binary(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if (!f) throw std::runtime_error("cannot read PBR cache: "+path);
    const std::streamsize n=f.tellg();
    if (n<=0) throw std::runtime_error("empty PBR cache: "+path);
    f.seekg(0);
    std::vector<uint8_t> bytes((size_t)n);
    if (!f.read((char*)bytes.data(),n)) throw std::runtime_error("short PBR cache read: "+path);
    // image_to_rig stage caches are size-prefixed contiguous arrays.  Keep accepting raw blobs as
    // well so this helper remains useful for a deliberately exported cache, but never let the
    // header become a phantom PBR voxel.
    if (bytes.size() >= sizeof(size_t)) {
        size_t payload=0; std::memcpy(&payload,bytes.data(),sizeof(payload));
        if (payload == bytes.size()-sizeof(size_t))
            return std::vector<uint8_t>(bytes.begin()+sizeof(size_t),bytes.end());
    }
    return bytes;
}

// Matches texture_mesh_native's Trellis texturing frame exactly.
static void preprocess_mesh(std::vector<float>& v) {
    float mn[3]={1e30f,1e30f,1e30f}, mx[3]={-1e30f,-1e30f,-1e30f};
    for (size_t i=0;i<v.size();i+=3) for (int c=0;c<3;c++) { mn[c]=std::min(mn[c],v[i+c]); mx[c]=std::max(mx[c],v[i+c]); }
    float center[3], extent=0.f;
    for (int c=0;c<3;c++) { center[c]=.5f*(mn[c]+mx[c]); extent=std::max(extent,mx[c]-mn[c]); }
    float scale=.99999f/std::max(extent,1e-12f);
    for (size_t i=0;i<v.size();i+=3) { float x=(v[i]-center[0])*scale, y=(v[i+1]-center[1])*scale, z=(v[i+2]-center[2])*scale;
        v[i]=x; v[i+1]=-z; v[i+2]=y; }
}
static void restore_mesh_frame(std::vector<float>& v) {
    for (size_t i=0;i<v.size();i+=3) { float y=v[i+1],z=v[i+2]; v[i+1]=z; v[i+2]=-y; }
}

// High native dumps store PBR after their optional 3-D outlier cleanup.  The marker lets a LOD
// rebake preserve that exact material instead of applying a second, non-idempotent median pass.
static float recorded_pbr_outlier_threshold(const std::string& pdir) {
    std::ifstream f(pdir+"/native_pbr_outlier_threshold.txt");
    float threshold=-1.f;
    if (f) f >> threshold;
    return threshold;
}

// The PBR voxel coordinates are integers on a grid-`resolution` lattice, and `resolution` is the
// ONLY thing that maps a normalised mesh position back onto them ((p+0.5)*resolution).  It is a
// property of the cache, not a caller preference: image_to_rig writes it next to the volume as a
// size-prefixed int32 (`resolution.bin`) when it stages the cache.  Consuming a grid-1536 cache as
// if it were grid-1024 shrinks every sample by 2/3 and silently bakes an almost entirely empty
// atlas, so read it rather than assume it.  Returns 0 when the cache predates the tag.
static int recorded_pbr_lattice_resolution(const std::string& dir) {
    std::ifstream f(dir+"/resolution.bin",std::ios::binary|std::ios::ate);
    if (!f) return 0;
    const std::streamsize n=f.tellg();
    if (n!=(std::streamsize)(sizeof(size_t)+sizeof(int32_t))) return 0;
    f.seekg(0);
    size_t payload=0; int32_t value=0;
    if (!f.read((char*)&payload,sizeof(payload)) || payload!=sizeof(int32_t)) return 0;
    if (!f.read((char*)&value,sizeof(value))) return 0;
    return value>0 ? (int)value : 0;
}

int main(int argc,char**argv) {
    std::string mesh_path,pdir,pbr_cache,out,atlas_out,unwrap="reference",bake_mode="projection"; int resolution=512,texsize=2048,decimate=0;
    bool resolution_requested=false;
    for (int i=1;i<argc;i++) { std::string a=argv[i];
        if(a=="--mesh"&&i+1<argc) mesh_path=argv[++i];
        else if(a=="--pbr-dir"&&i+1<argc) pdir=argv[++i];
        else if(a=="--pbr-cache"&&i+1<argc) pbr_cache=argv[++i];
        else if(a=="--out"&&i+1<argc) out=argv[++i];
        else if(a=="--resolution"&&i+1<argc) { resolution=atoi(argv[++i]); resolution_requested=true; }
        else if(a=="--texsize"&&i+1<argc) texsize=atoi(argv[++i]);
        else if(a=="--decimate"&&i+1<argc) decimate=atoi(argv[++i]);
        else if(a=="--unwrap"&&i+1<argc) unwrap=argv[++i];
        else if(a=="--atlas-out"&&i+1<argc) atlas_out=argv[++i];
        else if(a=="--bake"&&i+1<argc) bake_mode=argv[++i];
        else { usage(); return 2; }
    }
    // "volume-direct" is accepted as an explicit alias for the existing default projection route.
    if (bake_mode=="volume-direct") bake_mode="projection";
    if (bake_mode!="projection" && bake_mode!="volume-trilinear") { usage(); return 2; }
    // The lattice is whatever the generator used (Pixal's HR cascade is `grid = resolution/16`, so
    // 1024 and 1536 are both real production settings).  Accept the same shape image_to_rig accepts
    // -- a positive multiple of 16 -- instead of a hardcoded pair that cannot express 1536.
    if(mesh_path.empty()||(pdir.empty()==pbr_cache.empty())||out.empty()||resolution<=0||resolution%16!=0||texsize<=0 ||
       (unwrap!="production" && unwrap!="reference")) { usage(); return 2; }
    if (atlas_out.empty()) atlas_out=out+".atlas.png";
    try {
        // FAIL CLOSED on a lattice mismatch.  A cache that records its own lattice is authoritative:
        // adopt it when the caller said nothing, and refuse rather than bake when the caller
        // disagrees.  Silently honouring the caller here is exactly what produced an 88%-unpainted
        // high atlas from a grid-1536 cache requested at 1024.
        const std::string lattice_dir = pbr_cache.empty() ? pdir : pbr_cache;
        const int cache_resolution = recorded_pbr_lattice_resolution(lattice_dir);
        if (cache_resolution > 0) {
            if (!resolution_requested) {
                resolution = cache_resolution;
                std::printf("[native-rebake] PBR lattice resolution %d adopted from %s/resolution.bin\n",
                            resolution, lattice_dir.c_str());
            } else if (resolution != cache_resolution) {
                throw std::runtime_error("PBR lattice mismatch: cache "+lattice_dir+" is on the grid-"+
                                         std::to_string(cache_resolution)+" lattice but --resolution "+
                                         std::to_string(resolution)+" was requested; the voxel coordinates "
                                         "would be sampled in the wrong coordinate space");
            }
        } else if (!resolution_requested) {
            throw std::runtime_error("PBR cache "+lattice_dir+" has no resolution.bin lattice tag and no "
                                     "--resolution was given; refusing to guess the voxel coordinate space");
        }
        glb::Mesh mesh; if(!glb::read_glb(mesh_path.c_str(),mesh)) throw std::runtime_error("cannot read mesh: "+mesh_path);
        std::vector<float> pbr;
        std::vector<int32_t> coords;
        // A material-cache run has two geometry representations: `refined.glb` is the clean
        // delivery surface, while `coarse.glb` is the original Pixal surface on which the sparse
        // PBR volume was generated. Keep the latter as a material shell so later geometry cleanup
        // does not make the baked texture slide along the surface.
        std::vector<float> shell_verts, shell_attr;
        std::vector<int64_t> shell_faces;
        const bool native_retexture_frame=!pdir.empty();
        if (native_retexture_frame) {
            NpyArray pf=npy_load(pdir+"/native_pbr_feats.npy"), pc=npy_load(pdir+"/native_pbr_coords.npy");
            if(pf.descr!="<f4" || pf.shape.size()!=2 || pf.shape[1]!=6 || pc.shape.size()!=2 || pc.shape[1]!=4)
                throw std::runtime_error("invalid native PBR dump layout");
            int N=(int)pf.shape[0]; if((int)pc.shape[0]!=N) throw std::runtime_error("PBR feature/coordinate count mismatch");
            pbr.assign(pf.f32(),pf.f32()+(size_t)N*6);
            coords.resize((size_t)N*4);
            if(pc.descr=="<i4") std::memcpy(coords.data(),pc.i32(),coords.size()*sizeof(int32_t));
            else if(pc.descr=="<f4") for(size_t i=0;i<coords.size();i++) coords[i]=(int32_t)std::lround(pc.f32()[i]);
            else throw std::runtime_error("PBR coordinates must be int32 or float32");
        } else {
            // image_to_rig's cached Pixal M6 PBR volume is raw f32/i32, in the same canonical
            // frame as its refined GLB.  Reusing it avoids an unvalidated arbitrary-mesh
            // shape-encoder round trip; do not apply texture_mesh_native's frame transform here.
            std::vector<uint8_t> pb=load_binary(pbr_cache+"/pbr_feats.bin");
            std::vector<uint8_t> cb=load_binary(pbr_cache+"/pbr_coords.bin");
            if(pb.size()%(6*sizeof(float)) || cb.size()%(4*sizeof(int32_t)))
                throw std::runtime_error("invalid image_to_rig PBR cache layout");
            const size_t N=pb.size()/(6*sizeof(float));
            if(cb.size()/(4*sizeof(int32_t))!=N) throw std::runtime_error("PBR cache feature/coordinate count mismatch");
            pbr.resize(N*6); coords.resize(N*4);
            std::memcpy(pbr.data(),pb.data(),pb.size()); std::memcpy(coords.data(),cb.data(),cb.size());
            std::printf("[native-rebake] using direct Pixal image-to-rig PBR cache: %zu voxels\\n",N);
            glb::Mesh shell;
            const std::string shell_path=pbr_cache+"/coarse.glb";
            if (glb::read_glb(shell_path.c_str(),shell) && !shell.verts.empty() && !shell.faces.empty()) {
                shell_verts=std::move(shell.verts);
                shell_faces=std::move(shell.faces);
                std::printf("[native-rebake] retained Pixal material shell: %zu v / %zu f\\n",
                            shell_verts.size()/3,shell_faces.size()/3);
            } else {
                std::printf("[native-rebake] no usable coarse.glb; direct-volume compatibility bake\\n");
            }
        }
        const int N=(int)pbr.size()/6;
        // Second, tag-independent guard: every voxel index must be addressable on the lattice we are
        // about to sample.  A cache generated on a finer lattice than requested overflows this bound,
        // so an untagged legacy cache still cannot be consumed in the wrong coordinate space.
        {
            int32_t coord_max=-1;
            for (size_t i=0;i<coords.size();i++) coord_max=std::max(coord_max,coords[i]);
            if (coord_max >= resolution)
                throw std::runtime_error("PBR lattice mismatch: voxel coordinate "+std::to_string(coord_max)+
                                         " does not fit the requested grid-"+std::to_string(resolution)+
                                         " lattice; this volume was generated at a finer resolution");
            std::printf("[native-rebake] PBR lattice: grid-%d, max voxel coordinate %d\n",resolution,(int)coord_max);
        }
        // Keep a CPU LOD/rebake materially consistent with the direct-reference high bake.
        // The default is deliberately applied before the filter; TEX_PBR_OUTLIER=0 opt outs.
        const bool reference_unwrap = unwrap=="reference";
        if (reference_unwrap) setenv("TEX_PBR_OUTLIER", "0.10", 0);
        if (const char* f=std::getenv("TEX_PBR_OUTLIER")) {
            float threshold=(float)atof(f);
            int passes=1;
            if (const char* p=std::getenv("TEX_PBR_OUTLIER_PASSES")) passes=std::max(1,atoi(p));
            float recorded=native_retexture_frame ? recorded_pbr_outlier_threshold(pdir) : -1.f;
            if (threshold>0.f && std::fabs(recorded-threshold)<1e-6f) {
                std::printf("[native-rebake] PBR RGB outlier cleanup threshold=%.3f passes=%d: already applied by high material dump\n",threshold,passes);
            } else {
                size_t changed=texatlas::filter_pbr_rgb_outliers(pbr,coords,threshold,passes);
                std::printf("[native-rebake] PBR RGB outlier cleanup threshold=%.3f passes=%d: %zu / %d voxel replacements\n",threshold,passes,changed,N);
            }
        }
        // A direct texture_mesh_native dump is still in the shape-encoder frame and needs the
        // historical preprocess/restore pair.  In contrast, image_to_rig's PBR cache and the
        // native narrow-band mesh are already in the canonical Pixal frame.  Applying only the
        // restore half to that cache path silently rotates the delivered GLB (and makes a clean
        // mesh appear to have a texture alignment problem).
        if (native_retexture_frame) preprocess_mesh(mesh.verts);
        // `reference` removes the native pre-cluster stage and gives xatlas the same chart
        // parameters used by Pixal3D's CuMesh unwrap.  It intentionally retains the native
        // rasterizer and PBR sampling: this is an isolated, CPU-only unwrap/bake A/B.
        if (reference_unwrap) {
            // CPU rebakes must keep the high native reference bake's coverage, seam, and
            // normal behaviour so high/medium/low differ only by their intended LOD budget.
            // An explicit environment A/B setting still wins.
            auto bake_default=[](const char* key, const char* value) { setenv(key, value, 0); };
            setenv("ATL_PYREF_XATLAS", "1", 1);
            bake_default("TEX_RASTER_SS", "2");
            bake_default("TEX_TOPO_NORMALS", "1");
            bake_default("TEX_TELEA_INPAINT", "1");
            bake_default("TEX_TELEA_RADIUS", "4");
            bake_default("TEX_INPAINT_ITERS", "16");
            bake_default("TEX_FILL_BACKGROUND", "1");
            std::printf("[native-rebake] reference unwrap: direct mesh xatlas, Pixal3D chart settings\\n");
        }
        const bool cache_reproject=!native_retexture_frame && !shell_verts.empty() && !shell_faces.empty();
        if (cache_reproject) {
            // Keep the original coarse shell available as a guard for sparse-volume samples.
            // Direct-volume sampling preserves the generated PBR detail; the shell supplies a
            // finite, aligned fallback only where the refined target falls outside that volume.
            const int C=6, SV=(int)(shell_verts.size()/3);
            shell_attr.resize((size_t)SV*C);
            texgs::VolIndex vol(coords.data(),N,4,1);
            #pragma omp parallel for schedule(dynamic,4096)
            for (int i=0;i<SV;i++) {
                const float q0=(shell_verts[(size_t)i*3]+.5f)*resolution;
                const float q1=(shell_verts[(size_t)i*3+1]+.5f)*resolution;
                const float q2=(shell_verts[(size_t)i*3+2]+.5f)*resolution;
                texgs::sample_one(vol,pbr.data(),C,q0,q1,q2,&shell_attr[(size_t)i*C],2);
            }
            if (bake_mode=="volume-trilinear") {
                // Python o_voxel.postprocess.to_glb parity.  Python bakes each atlas texel by (1) BVH
                // unsigned_distance snapping the rasterised texel position back onto the ORIGINAL
                // high-res generation mesh, then (2) grid_sample_3d TRILINEAR of the sparse PBR volume
                // at that on-surface point, then (3) cv2.inpaint TELEA of the UV gutter.  The coarse
                // shell IS that generation surface, so the historical route snapped onto the coarse
                // shell (closest-point-on-triangle) and trilinear-sampled the volume there.
                //
                // CRISPNESS FIX (2026-07-24): that snap composes TWO closest-point projections
                // (refined-mesh texel -> coarse shell -> volume). The refined delivery surface is a
                // DIFFERENT surface from the coarse shell the volume was decoded on, so at a clothing
                // boundary the closest-point onto the coarse shell SLIDES across the boundary into the
                // neighbouring region -> the two materials blend -> smeared tie/collar/skirt edges.
                // Python does not have this because its textured mesh IS the coarse generation mesh
                // (self-consistent snap). To match Python's crispness we sample the volume at the
                // texel's OWN rasterised position on the mesh being textured (volume-direct: one
                // projection, no cross-surface slide), keeping the coarse shell only as the guard for
                // the ~0.3% of texels the volume has no data near. fallback_r=8 (call site) makes the
                // direct read robust to the refined surface sitting a few voxels off the sparse shell.
                // RP_TRILINEAR_SNAP=1 restores the old coarse-shell snap for A/B.
                unsetenv("RP_ATTR");
                // NORMAL-MARCH (2026-07-24, the fringe-speckle fix): volume-direct at the refined
                // texel reads the coarse grid-`gs` PBR shell in its FRINGE (the refined surface sits a
                // few voxels off), so partial trilinear support flips texel-to-texel -> speckle grain +
                // red splotches (v8). Marching along the texel normal onto the volume iso-surface reads
                // at FULL support with no lateral slide -> clean AND crisp on the refined geometry.
                // RP_TRILINEAR_SNAP=1 restores the old coarse-shell snap and RP_VOLUME_DIRECT=1 the v8
                // direct read, both for A/B.
                if(!std::getenv("RP_TRILINEAR_SNAP")) {
                    if (std::getenv("RP_VOLUME_DIRECT")) setenv("RP_DIRECT","1",1);
                    else setenv("RP_NORMAL_MARCH","1",1);
                }
                // RP_FRONTDOT=-1 disables front-face rejection on the guard snap so it takes the
                // GLOBALLY closest triangle exactly like Python's BVH.
                setenv("RP_FRONTDOT","-1",0);
                // Python o_voxel.to_glb stores the RAW volume baseColor (np.clip(attrs*255)); it does
                // NOT apply an sRGB OETF.  The default projection bake encodes sRGB, which is a concave
                // per-channel curve that compresses (max-min)/max -> ~24% desaturation vs Python's
                // stored atlas.  For Python parity keep the volume value raw (overridable).
                setenv("ATL_BASECOLOR_SRGB","0",0);
                setenv("TEX_TELEA_INPAINT","1",0);
                setenv("TEX_TELEA_RADIUS","3",0);
                // Python fills EVERY invalid texel (cv2.inpaint over the whole ~mask), so no chart is
                // bordered by unpainted background.  The native production route otherwise leaves the
                // inter-chart gutter black; with many small charts that black bleeds into chart edges
                // under mip/bilinear sampling and renders as the wireframe seam web.  Fill the gutter
                // by nearest-valid dilation (each gutter texel takes its own chart's edge colour) so
                // the seams disappear -- this is what closes the gap to Python's clean surface.
                setenv("TEX_FILL_BACKGROUND","1",0);
                std::printf("[native-rebake] cache sampling: volume-trilinear (Python to_glb parity: volume-direct at refined-mesh texel [crisp boundaries] + coarse-shell guard + RAW baseColor/no-sRGB + TELEA inpaint + background fill)\\n");
            } else if (std::getenv("TEX_REBAKE_CACHE_SHELL_ATTR")) {
                setenv("RP_ATTR","1",1); unsetenv("RP_DIRECT");
                std::printf("[native-rebake] cache sampling: shell-attribute reproject (diagnostic A/B)\\n");
            } else {
                unsetenv("RP_ATTR"); setenv("RP_DIRECT","1",1);
                std::printf("[native-rebake] cache sampling: volume-direct with coarse-shell fallback (projection default)\\n");
            }
        }
        texatlas::BakedTexture baked=cache_reproject
            ? texatlas::bake(mesh.verts,mesh.faces,pbr,coords,resolution,texsize,decimate,
                             reference_unwrap ? 0 : 4,true,8,!reference_unwrap,55.f,
                             &shell_verts,&shell_faces,&shell_attr,true)
            : texatlas::bake(mesh.verts,mesh.faces,pbr,coords,resolution,texsize,decimate,
                             reference_unwrap ? 0 : 4,true,8,!reference_unwrap);
        if (native_retexture_frame) {
            restore_mesh_frame(baked.verts);
            restore_mesh_frame(baked.normals);
        }
        if(!glb::write_glb_textured(out.c_str(),baked.verts,baked.normals,baked.uvs,baked.faces,
                                    baked.base_color,baked.metal_rough,baked.tw,baked.th))
            throw std::runtime_error("could not write output GLB");
        if(!stbi_write_png(atlas_out.c_str(),baked.tw,baked.th,4,baked.base_color.data(),baked.tw*4))
            throw std::runtime_error("could not write baseColor atlas: "+atlas_out);
        if(!texatlas::write_quality_report(out+".texture-qc.txt",baked))
            throw std::runtime_error("could not write texture quality report: "+out+".texture-qc.txt");
        std::printf("[native-rebake] DONE: %s (%d charts, %dx%d atlas)\\n",out.c_str(),baked.chart_count,baked.tw,baked.th);
        std::printf("[native-rebake] baseColor atlas: %s\\n",atlas_out.c_str());
        std::printf("[native-rebake] texture QC: %s.texture-qc.txt\\n",out.c_str());
    } catch(const std::exception&e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
    return 0;
}
