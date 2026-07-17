// rig_score.cpp — quantitative skeleton-quality scorer for a rigged GLB. Geometry-agnostic heuristic
// used to (a) rank rig outputs (best-of-N selection) and (b) judge any decode fix vs the Python baseline.
// CPU only, no ggml — reuses glb_reader.hpp (JSON parser + read_glb for the mesh bbox).
//
//   ./rig_score <rigged.glb> [expected_joints]
//
// Extracts joints (world positions from skins[0].joints node transforms) + parent forest (node
// hierarchy), and the mesh bbox, then reports:
//   J          joint count
//   roots      # parentless joints (clean rig = 1)
//   leaves     # joints with no child joint (biped ≈ head+2hands+2feet ≈ 4-8)
//   depth      max parent-chain depth (spine+limb ⇒ ≥ ~4)
//   symmetry   fraction of joints with a bilateral mirror partner (max over the X / Z mirror planes)
//   coverage   min over axes of joint-extent / mesh-extent (catches "clustered in one region")
//   structure  blend of root/leaf/depth plausibility
//   TOTAL      0.40*symmetry + 0.30*coverage + 0.30*structure   (0..1; higher = more biped-like)
// Heuristic, not ground truth — owner judges by RENDER; this just makes A/B/best-of-N objective.
#include "glb_reader.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <cstdlib>

using std::vector; using Mat4 = std::array<float,16>;  // column-major (glTF)

static Mat4 mat_identity() { Mat4 m{}; m[0]=m[5]=m[10]=m[15]=1.f; return m; }
static Mat4 mat_mul(const Mat4&a, const Mat4&b){ // a*b, column-major
    Mat4 c{};
    for(int col=0;col<4;col++) for(int row=0;row<4;row++){
        float s=0; for(int k=0;k<4;k++) s+=a[k*4+row]*b[col*4+k];
        c[col*4+row]=s;
    }
    return c;
}
static Mat4 node_local(const glb::detail::JVal& n){
    if(const glb::detail::JVal* mm=n.find("matrix")){ Mat4 m{}; for(int i=0;i<16&&i<(int)mm->arr.size();i++) m[i]=(float)mm->arr[i].num; return m; }
    Mat4 T=mat_identity(), R=mat_identity(), S=mat_identity();
    if(const glb::detail::JVal* t=n.find("translation")){ T[12]=(float)t->arr[0].num; T[13]=(float)t->arr[1].num; T[14]=(float)t->arr[2].num; }
    if(const glb::detail::JVal* q=n.find("rotation")){ float x=q->arr[0].num,y=q->arr[1].num,z=q->arr[2].num,w=q->arr[3].num;
        R[0]=1-2*(y*y+z*z); R[1]=2*(x*y+z*w);   R[2]=2*(x*z-y*w);
        R[4]=2*(x*y-z*w);   R[5]=1-2*(x*x+z*z); R[6]=2*(y*z+x*w);
        R[8]=2*(x*z+y*w);   R[9]=2*(y*z-x*w);   R[10]=1-2*(x*x+y*y); }
    if(const glb::detail::JVal* s=n.find("scale")){ S[0]=(float)s->arr[0].num; S[5]=(float)s->arr[1].num; S[10]=(float)s->arr[2].num; }
    return mat_mul(T, mat_mul(R,S));
}

int main(int argc,char**argv){
    if(argc<2){ std::fprintf(stderr,"usage: %s <rigged.glb> [expected_joints]\n",argv[0]); return 2; }
    const char* path=argv[1];
    int expected = argc>2 ? atoi(argv[2]) : -1;

    // ---- read the JSON chunk for the skeleton ----
    FILE* f=fopen(path,"rb"); if(!f){ std::fprintf(stderr,"cannot open %s\n",path); return 2; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    vector<uint8_t> buf(fsz); if(fread(buf.data(),1,fsz,f)!=(size_t)fsz){ fclose(f); return 2; } fclose(f);
    auto rd32=[&](size_t o){ uint32_t v; memcpy(&v,buf.data()+o,4); return v; };
    if(rd32(0)!=0x46546C67u){ std::fprintf(stderr,"bad glb magic\n"); return 2; }
    const char* json=nullptr; size_t jlen=0; uint32_t total=rd32(8); size_t off=12;
    while(off+8<=total){ uint32_t clen=rd32(off),ctype=rd32(off+4); size_t cs=off+8;
        if(ctype==0x4E4F534Au){ json=(const char*)(buf.data()+cs); jlen=clen; } off=cs+clen; }
    if(!json){ std::fprintf(stderr,"no JSON chunk\n"); return 2; }
    glb::detail::JVal root; if(!glb::detail::JParser(json,jlen).parse(root)||!root.is_obj()){ std::fprintf(stderr,"json parse failed\n"); return 2; }

    const glb::detail::JVal* nodes=root.find("nodes"); const glb::detail::JVal* skins=root.find("skins");
    if(!nodes||!nodes->is_arr()||!skins||!skins->is_arr()||skins->arr.empty()){
        std::fprintf(stderr,"%s: no skins/nodes (not a rigged glb)\n",path); printf("%s TOTAL=0.000 (no skeleton)\n",path); return 0; }
    const glb::detail::JVal* jl=skins->arr[0].find("joints");
    if(!jl||!jl->is_arr()){ std::fprintf(stderr,"no skin joints\n"); return 2; }
    int N=(int)nodes->arr.size();
    int J=(int)jl->arr.size();

    // node->parent (from children arrays), then world transforms
    vector<int> node_parent(N,-1);
    for(int i=0;i<N;i++){ const glb::detail::JVal* ch=nodes->arr[i].find("children"); if(ch&&ch->is_arr())
        for(auto& c:ch->arr){ int ci=(int)c.num; if(ci>=0&&ci<N) node_parent[ci]=i; } }
    vector<Mat4> world(N); vector<char> have(N,0);
    std::function<Mat4(int)> W=[&](int i)->Mat4{ if(have[i]) return world[i];
        Mat4 L=node_local(nodes->arr[i]); world[i]= (node_parent[i]>=0)? mat_mul(W(node_parent[i]),L):L; have[i]=1; return world[i]; };

    vector<int> jointNode(J); vector<int> nodeJoint(N,-1);
    for(int k=0;k<J;k++){ jointNode[k]=(int)jl->arr[k].num; nodeJoint[jointNode[k]]=k; }
    vector<std::array<float,3>> P(J);
    for(int k=0;k<J;k++){ Mat4 m=W(jointNode[k]); P[k]={m[12],m[13],m[14]}; }
    // skeleton parents (walk node parents to the first ancestor that is also a joint)
    vector<int> parent(J,-1);
    for(int k=0;k<J;k++){ int p=node_parent[jointNode[k]]; while(p>=0&&nodeJoint[p]<0) p=node_parent[p]; parent[k]= p>=0?nodeJoint[p]:-1; }

    // ---- mesh bbox (for coverage) ----
    glb::Mesh mesh; float mext[3]={0,0,0}; bool have_mesh=glb::read_glb(path,mesh);
    if(have_mesh && !mesh.verts.empty()){ float lo[3]={1e30f,1e30f,1e30f},hi[3]={-1e30f,-1e30f,-1e30f};
        for(size_t i=0;i+2<mesh.verts.size();i+=3) for(int d=0;d<3;d++){ float v=mesh.verts[i+d]; lo[d]=std::min(lo[d],v); hi[d]=std::max(hi[d],v); }
        for(int d=0;d<3;d++) mext[d]=hi[d]-lo[d]; }

    // ---- metrics ----
    int roots=0; for(int k=0;k<J;k++) if(parent[k]<0) roots++;
    vector<int> nchild(J,0); for(int k=0;k<J;k++) if(parent[k]>=0) nchild[parent[k]]++;
    int leaves=0; for(int k=0;k<J;k++) if(nchild[k]==0) leaves++;
    // max_children = the "fan" defect signal. A real biped branches at most ~5 (pelvis->spine+2legs,
    // chest->neck+2arms); Python's beam-sample ceiling on gilly was 5. The C++ sampled beam OCCASIONALLY
    // emits a head-fan (one joint with 16-26+ children) that bilateral-symmetry MISSES (a roughly radial
    // fan reflects onto itself). Score it explicitly so best-of-N rejects fanned rigs. (See the maxfan
    // sweep: clean seeds maxfan<=6, fanned/runaway seeds 16/26/143.)
    int max_children=0; for(int k=0;k<J;k++) max_children=std::max(max_children,nchild[k]);
    int max_depth=0; for(int k=0;k<J;k++){ int d=0,c=k,guard=0; while(parent[c]>=0&&guard++<J){ d++; c=parent[c]; } max_depth=std::max(max_depth,d); }

    // joints bbox + centroid
    float jlo[3]={1e30f,1e30f,1e30f},jhi[3]={-1e30f,-1e30f,-1e30f},cen[3]={0,0,0};
    for(int k=0;k<J;k++) for(int d=0;d<3;d++){ jlo[d]=std::min(jlo[d],P[k][d]); jhi[d]=std::max(jhi[d],P[k][d]); cen[d]+=P[k][d]/J; }
    float jext[3]; for(int d=0;d<3;d++) jext[d]=jhi[d]-jlo[d];
    float meshdiag = have_mesh ? std::sqrt(mext[0]*mext[0]+mext[1]*mext[1]+mext[2]*mext[2]) : std::sqrt(jext[0]*jext[0]+jext[1]*jext[1]+jext[2]*jext[2]);

    // coverage = min axis of jointExtent/meshExtent (clamped)
    float coverage=1.f;
    if(have_mesh) for(int d=0;d<3;d++){ float r= mext[d]>1e-6f ? jext[d]/mext[d] : 1.f; coverage=std::min(coverage,std::min(1.f,r)); }

    // symmetry: bilateral mirror match across each axis' centroid plane. COLLAPSE-HOLE FIX: a tight
    // joint cluster (or a flat fan) used to score 1.0 because every mirrored point lands near SOME other
    // joint — trivial self-reflection. Now off-plane joints must find a DISTINCT one-to-one partner on
    // the OPPOSITE side of the plane (greedy nearest; used[] forbids many-to-one), and the score is
    // GATED by how many genuine bilateral PAIRS exist: a blob has ~0 straddling pairs -> gate -> ~0, so
    // it can no longer win on degenerate symmetry. On-plane (spine) joints still count (self-symmetric),
    // but only when real pairs establish the mirror plane. Pick the axis maximizing match*extent-fraction.
    float tol = 0.06f*meshdiag; if(tol<=0) tol=0.06f;
    float jdiag=std::sqrt(jext[0]*jext[0]+jext[1]*jext[1]+jext[2]*jext[2]); if(jdiag<=0) jdiag=1.f;
    auto sym_for=[&](int ax, int* out_pairs)->float{
        vector<char> used(J,0); int on_plane=0, paired=0, pairs=0;
        for(int k=0;k<J;k++){
            if(used[k]) continue;
            float offk=P[k][ax]-cen[ax];
            if(std::fabs(offk)<=tol){ used[k]=1; on_plane++; continue; }   // spine / on-axis joint
            std::array<float,3> r=P[k]; r[ax]=2*cen[ax]-r[ax];             // mirror across the plane
            int bestj=-1; float bestd=tol;
            for(int j=0;j<J;j++){ if(j==k||used[j]) continue;
                float offj=P[j][ax]-cen[ax]; if(offk*offj>=0) continue;     // partner must straddle the plane
                float dx=r[0]-P[j][0],dy=r[1]-P[j][1],dz=r[2]-P[j][2]; float d=std::sqrt(dx*dx+dy*dy+dz*dz);
                if(d<bestd){ bestd=d; bestj=j; } }
            if(bestj>=0){ used[k]=1; used[bestj]=1; paired+=2; pairs++; }   // distinct one-to-one pair
        }
        if(out_pairs) *out_pairs=pairs;
        float raw=(float)(on_plane+paired)/std::max(1,J);
        // gate=1 once >= ~12% of joints form genuine straddling pairs (a real biped easily clears this);
        // a cluster/fan with no pairs gates to 0 no matter how it self-reflects.
        float gate=std::min(1.f, (float)pairs/std::max(1.f,0.12f*(float)J));
        return raw*gate;
    };
    int px=0,py=0,pz=0;
    float sx=sym_for(0,&px), sy=sym_for(1,&py), sz=sym_for(2,&pz);
    float symmetry=sx; { float sa[3]={sx,sy,sz}; float bestw=-1.f;
        for(int a=0;a<3;a++){ float w=sa[a]*(jext[a]/jdiag); if(w>bestw){ bestw=w; symmetry=sa[a]; } } }

    // structure plausibility: single root + ANATOMICALLY plausible chain depth (penalize both too-shallow
    // collapse and runaway ghost chains like the 48-deep seed-3). Leaf count is NOT scored — a clean rig
    // with fingers/toes legitimately has many leaf joints.
    float root_s = roots==1 ? 1.f : 1.f/std::max(1,roots);
    auto depth_plaus=[&](int d)->float{ if(d>=4&&d<=16) return 1.f; if(d<4) return std::max(0.f,d/4.f);
                                        return std::max(0.f, 1.f-(d-16)/24.f); };   // ->0 at depth 40
    float depth_s=depth_plaus(max_depth);
    float structure=0.5f*root_s + 0.5f*depth_s;

    // fan penalty: 1.0 while max_children<=6 (a clean biped never exceeds this), decaying to 0 by >=12.
    // A head-fan / runaway is an unambiguous defect, so it multiplies the whole score (not a weighted term).
    float fan_s = max_children<=6 ? 1.f : std::max(0.f, 1.f - (float)(max_children-6)/6.f);

    float TOTAL = (0.50f*symmetry + 0.30f*coverage + 0.20f*structure) * fan_s;

    printf("%-40s J=%-3d", path, J);
    if(expected>0) printf(" (exp~%d)",expected);
    printf("  roots=%d leaves=%d depth=%d maxfan=%d  symmetry=%.3f coverage=%.3f structure=%.3f fan=%.2f  TOTAL=%.3f\n",
           roots,leaves,max_depth,max_children, symmetry,coverage,structure,fan_s, TOTAL);
    printf("    sub: root_s=%.2f depth_s=%.2f | sym_x=%.3f sym_y=%.3f sym_z=%.3f | jext=[%.2f %.2f %.2f] mext=[%.2f %.2f %.2f] mesh=%s\n",
           root_s,depth_s, sx,sy,sz, jext[0],jext[1],jext[2], mext[0],mext[1],mext[2], have_mesh?"yes":"NO");
    return 0;
}
