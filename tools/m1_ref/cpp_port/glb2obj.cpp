#include <cstdio>
#include "glb_reader.hpp"
int main(int argc,char**argv){ if(argc<3){printf("usage: glb2obj in.glb out.obj\n");return 1;}
 glb::Mesh m; if(!glb::read_glb(argv[1],m)){printf("read fail\n");return 1;}
 FILE*f=fopen(argv[2],"w"); size_t nv=m.verts.size()/3,nf=m.faces.size()/3;
 for(size_t i=0;i<nv;i++)fprintf(f,"v %.9g %.9g %.9g\n",m.verts[i*3],m.verts[i*3+1],m.verts[i*3+2]);
 for(size_t t=0;t<nf;t++)fprintf(f,"f %lld %lld %lld\n",(long long)m.faces[t*3]+1,(long long)m.faces[t*3+1]+1,(long long)m.faces[t*3+2]+1);
 fclose(f); printf("wrote %s (%zu v/%zu f)\n",argv[2],nv,nf); return 0;}
