// obj_fill_holes — close the residual boundary-loop holes Instant Meshes' extraction leaves in the
// retopo (degree 7-54 gaps it "didn't try to fill"). Reads a quad/tri OBJ (f a//n b//n c//n [d//n]),
// detects boundary edges (a directed edge with no opposite), traces oriented boundary loops, and
// fan-fills each loop (centroid vertex + triangle fan). Original quads are preserved verbatim (keeps
// the field-aligned edge flow for rigging); only the ~40 holes get tri patches. The normal-map bake
// from the dense mesh hides the patch shading.
//   build: g++ -O2 -std=c++17 obj_fill_holes.cpp -o obj_fill_holes
//   run:   ./obj_fill_holes in.obj out.obj [max_loop_deg=400]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>

struct V3 { double x,y,z; };

int main(int argc, char** argv){
    if(argc<3){ printf("usage: obj_fill_holes in.obj out.obj [max_loop_deg=400]\n"); return 1; }
    int max_deg = argc>3 ? atoi(argv[3]) : 400;

    std::vector<V3> V;
    std::vector<std::vector<uint32_t>> faces;     // original polygons (0-based), preserved verbatim
    {
        FILE* f=fopen(argv[1],"r"); if(!f){printf("open %s failed\n",argv[1]);return 1;}
        char line[1024];
        while(fgets(line,sizeof line,f)){
            if(line[0]=='v'&&line[1]==' '){ V3 p; sscanf(line+2,"%lf %lf %lf",&p.x,&p.y,&p.z); V.push_back(p); }
            else if(line[0]=='f'&&line[1]==' '){
                std::vector<uint32_t> poly; char* s=line+2;
                while(*s){ while(*s==' '||*s=='\n')s++; if(!*s)break;
                    long idx=strtol(s,&s,10); if(idx==0)break; if(idx<0)idx=(long)V.size()+idx+1;
                    poly.push_back((uint32_t)(idx-1)); while(*s&&*s!=' '&&*s!='\n')s++; }
                if(poly.size()>=3) faces.push_back(poly);
            }
        }
        fclose(f);
    }
    printf("[in] %zu verts, %zu faces\n", V.size(), faces.size());

    // directed-edge map over triangulated faces (fan) to find boundary edges.
    std::unordered_set<uint64_t> dir; dir.reserve(faces.size()*6);
    auto dkey=[](uint32_t a,uint32_t b){ return (uint64_t)a<<32 | b; };
    for(auto&p:faces) for(size_t k=1;k+1<p.size();k++){
        uint32_t a=p[0],b=p[k],c=p[k+1];
        dir.insert(dkey(a,b)); dir.insert(dkey(b,c)); dir.insert(dkey(c,a));
    }
    // boundary directed edge = one whose reverse is absent. Build a->b successor map.
    // per-vertex list of outgoing boundary edges (a->b). Handles non-manifold junctions: a vertex with
    // 2+ outgoing boundary edges (bowtie) keeps ALL of them, so every loop through it can be traced by
    // CONSUMING edges one at a time (the old single-successor map dropped all but one → 300+ holes left
    // unfilled). This closes every closed boundary loop, bowties included.
    std::unordered_map<uint32_t,std::vector<uint32_t>> out; out.reserve(4096);
    size_t bnd=0;
    for(uint64_t e:dir){ uint32_t a=e>>32, b=(uint32_t)e; if(!dir.count(dkey(b,a))){ out[a].push_back(b); bnd++; } }
    printf("[bnd] %zu boundary directed edges\n", bnd);

    // trace loops by consuming edges. From each vertex with unused outgoing edges, walk picking any
    // unused edge, marking it used, until we return to the loop start (closed loop) or dead-end.
    std::unordered_map<uint32_t,size_t> usedCnt; usedCnt.reserve(out.size());
    size_t loops=0, filled=0, skipped=0, addtris=0;
    std::vector<std::array<uint32_t,3>> fill;
    for(auto&kv:out){
        uint32_t base=kv.first;
        while(usedCnt[base] < out[base].size()){
            // start a fresh loop at `base` consuming its next edge
            std::vector<uint32_t> loop; uint32_t cur=base; bool ok=false;
            while(true){
                auto& vec=out[cur]; size_t& u=usedCnt[cur];
                if(u>=vec.size()){ ok=false; break; }           // dead-end (open chain)
                uint32_t nxt=vec[u++]; loop.push_back(cur);
                if(nxt==base){ ok=true; break; }                // closed the loop
                cur=nxt;
                if(loop.size()>(size_t)max_deg+1){ ok=false; break; }
            }
            if(!ok || loop.size()<3){ skipped++; continue; }
        loops++;
        if((int)loop.size()>max_deg){ skipped++; continue; }
        // centroid vertex
        V3 c{0,0,0}; for(uint32_t vi:loop){ c.x+=V[vi].x; c.y+=V[vi].y; c.z+=V[vi].z; }
        c.x/=loop.size(); c.y/=loop.size(); c.z/=loop.size();
        uint32_t ci=(uint32_t)V.size(); V.push_back(c);
        // fan: loop is the boundary in CCW-of-hole order (a->b boundary). Triangle (b,a,center) so the
        // patch faces outward consistently with the boundary edge orientation.
        for(size_t i=0;i<loop.size();i++){ uint32_t a=loop[i], b=loop[(i+1)%loop.size()];
            fill.push_back({b,a,ci}); addtris++; }
        filled++;
        }   // while(unused edges at base)
    }       // for each base vertex
    printf("[fill] %zu loops, filled %zu, skipped %zu (>deg %d or open), added %zu tris\n",
           loops, filled, skipped, max_deg, addtris);

    FILE* o=fopen(argv[2],"w"); if(!o){printf("out open failed\n");return 1;}
    for(auto&p:V) fprintf(o,"v %g %g %g\n",p.x,p.y,p.z);
    for(auto&p:faces){ fprintf(o,"f"); for(uint32_t vi:p) fprintf(o," %u",vi+1); fprintf(o,"\n"); }
    for(auto&t:fill) fprintf(o,"f %u %u %u\n",t[0]+1,t[1]+1,t[2]+1);
    fclose(o);
    printf("[out] wrote %s: %zu verts, %zu faces (+%zu fill tris)\n", argv[2], V.size(), faces.size()+fill.size(), fill.size());
    return 0;
}
