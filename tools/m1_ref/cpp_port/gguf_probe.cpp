// Probe a GGUF: does ggml load it, and what types are the tensors? Build: ./build.sh gguf_probe
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <string>
int main(int argc, char** argv){
    if (argc<2){ printf("usage: gguf_probe <file.gguf> [tensor_name]\n"); return 1; }
    ggml_context* data=nullptr; gguf_init_params p{false,&data};
    gguf_context* g = gguf_init_from_file(argv[1], p);
    if (!g){ printf("gguf_init_from_file FAILED for %s\n", argv[1]); return 1; }
    int64_t nt = gguf_get_n_tensors(g);
    printf("loaded %s : %lld tensors\n", argv[1], (long long)nt);
    int q=0,f=0;
    for (int64_t i=0;i<nt;i++){
        const char* nm = gguf_get_tensor_name(g, i);
        ggml_tensor* t = ggml_get_tensor(data, nm);
        if (t->type==GGML_TYPE_F32) f++; else q++;
    }
    printf("  f32=%d  non-f32(quant)=%d\n", f, q);
    if (argc>=3){ ggml_tensor* t=ggml_get_tensor(data, argv[2]);
        if(t) printf("  %s: type=%s ne=[%lld,%lld,%lld,%lld]\n", argv[2], ggml_type_name(t->type),
            (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3]);
        else printf("  tensor %s not found\n", argv[2]); }
    gguf_free(g); ggml_free(data);
    return 0;
}
