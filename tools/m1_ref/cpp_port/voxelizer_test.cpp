// voxelizer_test.cpp — validate vox::mesh_to_flexible_dual_grid against the o_voxel golden.
// Loads the EXACT preprocessed verts/faces the python capture used (verts_pre.npy/faces.npy in
// the golden dir) so the only variable is the voxelizer math, then diffs:
//   coords  : as a SET (IoU + recall + precision) — order differs (o_voxel hashmap order)
//   dual    : max-abs on matched cells
//   inter   : per-axis agreement % on matched cells
//
// build: ./build.sh voxelizer_test
#include "voxelizer.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// ---- minimal .npy reader (float32 / int32 / int8, C-order) ----
struct Npy { std::vector<size_t> shape; char dt; int esz; std::vector<char> raw;
    size_t numel() const { size_t n=1; for(auto s:shape)n*=s; return n; } };
static Npy npy_load(const std::string& path) {
    Npy a; FILE* f=fopen(path.c_str(),"rb"); if(!f){fprintf(stderr,"open fail %s\n",path.c_str());exit(1);}
    char magic[6]; if(fread(magic,1,6,f)!=6||memcmp(magic,"\x93NUMPY",6)){fprintf(stderr,"bad npy %s\n",path.c_str());exit(1);}
    unsigned char ver[2]; if(fread(ver,1,2,f)!=2){exit(1);}
    uint16_t hlen; if(fread(&hlen,2,1,f)!=1){exit(1);}
    std::string hdr(hlen,0); if(fread(&hdr[0],1,hlen,f)!=hlen){exit(1);}
    // descr
    size_t dp=hdr.find("descr"); dp=hdr.find(":",dp); std::string d=hdr.substr(hdr.find("'",dp)+1); d=d.substr(0,d.find("'"));
    if(d.find("f4")!=std::string::npos){a.dt='f';a.esz=4;}
    else if(d.find("i4")!=std::string::npos){a.dt='i';a.esz=4;}
    else if(d.find("i1")!=std::string::npos||d.find("b1")!=std::string::npos){a.dt='b';a.esz=1;}
    else if(d.find("i8")!=std::string::npos){a.dt='l';a.esz=8;}
    else {fprintf(stderr,"unsupported dtype %s in %s\n",d.c_str(),path.c_str());exit(1);}
    // shape
    size_t sp=hdr.find("shape"); size_t lp=hdr.find("(",sp), rp=hdr.find(")",lp);
    std::string sh=hdr.substr(lp+1,rp-lp-1);
    size_t pos=0; while(pos<sh.size()){ while(pos<sh.size()&&(sh[pos]==' '||sh[pos]==','))pos++;
        if(pos>=sh.size())break; size_t e=pos; while(e<sh.size()&&sh[e]>='0'&&sh[e]<='9')e++;
        if(e>pos)a.shape.push_back((size_t)atoll(sh.substr(pos,e-pos).c_str())); pos=e; }
    a.raw.resize(a.numel()*a.esz);
    if(fread(a.raw.data(),1,a.raw.size(),f)!=a.raw.size()){fprintf(stderr,"short read %s\n",path.c_str());exit(1);}
    fclose(f); return a;
}

static int64_t key3(int32_t x,int32_t y,int32_t z){ return vox::cell_key(x,y,z); }

// Keep the set/feature comparison auditable: when a full fixture disagrees,
// this optional dump lets an external NumPy comparator inspect the exact
// native cells rather than trying to infer them from aggregate percentages.
static void dump_npy(const std::string& path, const char* descr,
                     size_t rows, size_t cols, const void* data, size_t bytes) {
    std::string hdr = std::string("{'descr': '") + descr +
        "', 'fortran_order': False, 'shape': (" + std::to_string(rows) +
        ", " + std::to_string(cols) + "), }";
    const size_t preamble = 10; // magic + v1.0 version + uint16 header length
    const size_t padded = ((preamble + hdr.size() + 1 + 15) / 16) * 16;
    hdr.append(padded - preamble - hdr.size() - 1, ' ');
    hdr.push_back('\n');
    if (hdr.size() > 65535) { std::fprintf(stderr, "npy header too large\n"); std::exit(1); }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); std::exit(1); }
    const char magic[] = {'\x93','N','U','M','P','Y',1,0};
    const uint16_t hlen = (uint16_t)hdr.size();
    f.write(magic, sizeof(magic)); f.write((const char*)&hlen, sizeof(hlen));
    f.write(hdr.data(), (std::streamsize)hdr.size());
    f.write((const char*)data, (std::streamsize)bytes);
    if (!f) { std::fprintf(stderr, "short write %s\n", path.c_str()); std::exit(1); }
}

int main(int argc, char** argv) {
    std::string gdir = argc>1 ? argv[1] : "/mnt/hdd/pixal3d_tex/golden_voxel";
    int grid = argc>2 ? atoi(argv[2]) : 1024;

    Npy vp = npy_load(gdir+"/verts_pre.npy");   // [V,3] f32
    Npy fc = npy_load(gdir+"/faces.npy");        // [F,3] i4 (or i8)
    int V=(int)vp.shape[0], F=(int)fc.shape[0];
    const float* verts=(const float*)vp.raw.data();
    std::vector<int32_t> faces((size_t)F*3);
    if(fc.dt=='i'){ memcpy(faces.data(),fc.raw.data(),(size_t)F*3*4); }
    else if(fc.dt=='l'){ const int64_t* p=(const int64_t*)fc.raw.data(); for(size_t i=0;i<(size_t)F*3;i++)faces[i]=(int32_t)p[i]; }
    printf("loaded verts_pre [%d,3] faces [%d,3]\n",V,F);

    int gs[3]={grid,grid,grid};
    float amn[3]={-0.5f,-0.5f,-0.5f}, amx[3]={0.5f,0.5f,0.5f};
    vox::VoxelOut o = vox::mesh_to_flexible_dual_grid(verts,V,faces.data(),F,gs,amn,amx,1.0f,0.2f,1e-2f);
    printf("C++ voxelizer N=%d\n",o.N);
    if (const char* out = std::getenv("VOX_DUMP_DIR")) {
        const std::string root(out);
        dump_npy(root + "/native_voxel_coords.npy", "<i4", (size_t)o.N, 3,
                 o.coords.data(), o.coords.size() * sizeof(int32_t));
        dump_npy(root + "/native_voxel_dual_local.npy", "<f4", (size_t)o.N, 3,
                 o.dual.data(), o.dual.size() * sizeof(float));
        dump_npy(root + "/native_voxel_intersected.npy", "|i1", (size_t)o.N, 3,
                 o.intersected.data(), o.intersected.size() * sizeof(int8_t));
        std::printf("wrote native voxel dump: %s\n", root.c_str());
    }

    // load golden
    Npy gc=npy_load(gdir+"/coords.npy"), gd=npy_load(gdir+"/dual_vertices.npy"), gi=npy_load(gdir+"/intersected.npy");
    // Historical local captures stored [x,y,z]; the official encoder boundary
    // stores [batch,x,y,z].  Treat both as the same spatial fixture rather
    // than accidentally comparing batch/x/y against x/y/z.
    if (gc.shape.size()!=2 || (gc.shape[1]!=3 && gc.shape[1]!=4)) {
        std::fprintf(stderr, "golden coords must be [N,3] or [N,4], got rank=%zu width=%zu\n",
                     gc.shape.size(), gc.shape.size()>1 ? gc.shape[1] : 0); return 1;
    }
    const int gc_stride=(int)gc.shape[1], gc_xyz=gc_stride-3;
    int M=(int)gc.shape[0];
    const int32_t* gco=(const int32_t*)gc.raw.data();
    const float*   gdu=(const float*)gd.raw.data();
    const int8_t*  gin=(const int8_t*)gi.raw.data();
    printf("golden N=%d\n",M);

    // optional per-cell dump: env DUMPN dumps first N matched cells (ours vs golden)
    // golden map
    std::unordered_map<int64_t,int> gmap; gmap.reserve((size_t)M*2);
    auto gcoord = [&](int i, int a) -> int32_t { return gco[(size_t)i*gc_stride+gc_xyz+a]; };
    for(int i=0;i<M;i++) gmap.emplace(key3(gcoord(i,0),gcoord(i,1),gcoord(i,2)),i);
    // ours map
    std::unordered_map<int64_t,int> omap; omap.reserve((size_t)o.N*2);
    for(int i=0;i<o.N;i++) omap.emplace(key3(o.coords[i*3],o.coords[i*3+1],o.coords[i*3+2]),i);

    // SET diff
    long inter=0; for(auto&kv:omap) if(gmap.count(kv.first)) inter++;
    long uni=(long)omap.size()+(long)gmap.size()-inter;
    double iou=(double)inter/uni, recall=(double)inter/M, prec=(double)inter/o.N;
    printf("\n=== COORDS SET ===\n");
    printf("  golden %d, ours %d, intersection %ld\n",M,o.N,inter);
    printf("  IoU %.4f  recall %.4f  precision %.4f\n",iou,recall,prec);

    // dual maxabs + intersected agreement on MATCHED cells (compare ours vs golden by key)
    double dmax=0,dsum=0; long dn=0;
    long iagree=0,itot=0;
    for(int i=0;i<o.N;i++){
        auto it=gmap.find(key3(o.coords[i*3],o.coords[i*3+1],o.coords[i*3+2]));
        if(it==gmap.end()) continue;
        int gj=it->second;
        for(int a=0;a<3;a++){
            // golden dual is the GLOBAL normalized coord (coord+offset)/grid; recover in-cell offset.
            double goff = (double)gdu[(size_t)gj*3+a]*grid - (double)gcoord(gj,a);
            double e=std::fabs((double)o.dual[(size_t)i*3+a]-goff);
            dmax=std::max(dmax,e); dsum+=e; dn++;
            itot++; if(o.intersected[(size_t)i*3+a]==gin[(size_t)gj*3+a]) iagree++;
        }
    }
    printf("\n=== DUAL (matched cells) ===\n");
    printf("  maxabs %.5f  meanabs %.5f  (over %ld comps)\n",dmax,dsum/(dn?dn:1),dn);
    printf("\n=== INTERSECTED (matched cells) ===\n");
    printf("  agreement %.4f  (%ld/%ld)\n",(double)iagree/(itot?itot:1),iagree,itot);
    // per-axis confusion (ours vs golden)
    long c11[3]={0},c10[3]={0},c01[3]={0},c00[3]={0};
    for(int i=0;i<o.N;i++){ auto it=gmap.find(key3(o.coords[i*3],o.coords[i*3+1],o.coords[i*3+2])); if(it==gmap.end())continue; int gj=it->second;
        for(int a=0;a<3;a++){int oo=o.intersected[i*3+a],gg=gin[gj*3+a]; if(oo&&gg)c11[a]++; else if(oo&&!gg)c10[a]++; else if(!oo&&gg)c01[a]++; else c00[a]++;}}
    for(int a=0;a<3;a++) printf("  axis %d: both1=%ld ours1golden0=%ld ours0golden1=%ld both0=%ld\n",a,c11[a],c10[a],c01[a],c00[a]);
    // Diagnose coordinate convention for exact edge tests: find the native
    // flag translation with the highest overlap against the reference.
    for(int a=0;a<3;a++) {
        long best=-1; int bx=0,by=0,bz=0;
        for(int dz=-1;dz<=1;dz++) for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++) {
            long hit=0;
            for(int j=0;j<M;j++) if(gin[j*3+a]) {
                auto it=omap.find(key3(gcoord(j,0)+dx,gcoord(j,1)+dy,gcoord(j,2)+dz));
                if(it!=omap.end() && o.intersected[it->second*3+a]) hit++;
            }
            if(hit>best){best=hit;bx=dx;by=dy;bz=dz;}
        }
        printf("  axis %d: best native flag shift vs golden = [%d %d %d], overlap %ld/%ld\n",a,bx,by,bz,best,c11[a]+c01[a]);
    }

    if(getenv("DUMPN")){ int dn2=atoi(getenv("DUMPN")); int shown=0;
        for(int i=0;i<o.N && shown<dn2;i++){ auto it=gmap.find(key3(o.coords[i*3],o.coords[i*3+1],o.coords[i*3+2])); if(it==gmap.end())continue; int gj=it->second;
            printf("cell(%d,%d,%d) ours dual[%.4f %.4f %.4f] golden[%.4f %.4f %.4f] | ours int[%d%d%d] gold[%d%d%d]\n",
                o.coords[i*3],o.coords[i*3+1],o.coords[i*3+2],
                o.dual[i*3],o.dual[i*3+1],o.dual[i*3+2],
                gdu[gj*3]*grid-gcoord(gj,0), gdu[gj*3+1]*grid-gcoord(gj,1), gdu[gj*3+2]*grid-gcoord(gj,2),
                o.intersected[i*3],o.intersected[i*3+1],o.intersected[i*3+2], gin[gj*3],gin[gj*3+1],gin[gj*3+2]);
            shown++; } }

    // also: downsample //16 set IoU vs golden_69k (the encoder lattice) — the load-bearing metric
    Npy g69; bool has69=false;
    { FILE* t=fopen("/mnt/hdd/pixal3d_tex/golden_69k_1024/shape_slat_coords.npy","rb"); if(t){fclose(t);
        g69=npy_load("/mnt/hdd/pixal3d_tex/golden_69k_1024/shape_slat_coords.npy"); has69=true; } }
    if(has69){
        const int32_t* g=(const int32_t*)g69.raw.data(); int K=(int)g69.shape[0]; // [K,4] b,x,y,z
        std::unordered_map<int64_t,int> g64; for(int i=0;i<K;i++) g64.emplace(key3(g[i*4+1],g[i*4+2],g[i*4+3]),i);
        std::unordered_map<int64_t,char> o64;
        for(int i=0;i<o.N;i++) o64.emplace(key3(o.coords[i*3]/16,o.coords[i*3+1]/16,o.coords[i*3+2]/16),1);
        long in2=0; for(auto&kv:o64) if(g64.count(kv.first))in2++;
        long un2=(long)o64.size()+(long)g64.size()-in2;
        printf("\n=== DOWNSAMPLE //16 vs golden_69k @grid64 ===\n");
        printf("  golden %d, ours %zu, IoU %.4f recall %.4f\n",K,o64.size(),(double)in2/un2,(double)in2/K);
    }
    return 0;
}
