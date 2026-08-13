#include <cstdio>
#include "im_retopo.hpp"      // read_obj_tri (triangulates quads) + svae::Mesh
#include "glb_writer.hpp"
int main(int argc,char**argv){ if(argc<3){printf("usage: obj2glb in.obj out.glb\n");return 1;}
 svae::Mesh m; if(!imretopo::read_obj_tri(argv[1],m)){printf("read fail\n");return 1;}
 if(!glb::write_glb(argv[2],m.verts,m.faces)){printf("write fail\n");return 1;}
 printf("wrote %s (%d v/%d f)\n",argv[2],m.N,m.F); return 0;}
