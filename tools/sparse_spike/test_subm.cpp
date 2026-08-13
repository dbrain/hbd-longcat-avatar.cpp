// Validate the kernel-interface impl (launch_subm_conv_*) against goldens.
// Builds the rulebook on host, runs the fp32 implicit-GEMM, compares to the f64
// out_feats.npy golden within fp32 tolerance.
//
// CPU now:   g++ -O2 -std=c++17 test_subm.cpp -o test_subm && ./test_subm golden_model
// GPU later: nvcc ... -DUSE_CUDA test_subm.cpp sparse_subm_conv.cu (run vs same goldens)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

#include "npy.hpp"
#include "sparse_subm_conv_cpu.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
extern "C" void launch_subm_conv_cuda(const float*, const uint32_t*, const float*,
                                      const float*, float*, int, int, int, int, cudaStream_t);
static std::vector<float> run_cuda(const float* feats, const uint32_t* nmap,
                                   const float* weight, const float* bias,
                                   int N, int Cin, int Cout, int V, double& ms) {
    float *df, *dw, *db, *dout; uint32_t* dn;
    cudaMalloc(&df, (size_t)N*Cin*4);  cudaMalloc(&dn, (size_t)N*V*4);
    cudaMalloc(&dw, (size_t)V*Cin*Cout*4); cudaMalloc(&db, (size_t)Cout*4);
    cudaMalloc(&dout, (size_t)N*Cout*4);
    cudaMemcpy(df, feats, (size_t)N*Cin*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, nmap, (size_t)N*V*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, weight, (size_t)V*Cin*Cout*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias, (size_t)Cout*4, cudaMemcpyHostToDevice);
    for (int i = 0; i < 3; i++) launch_subm_conv_cuda(df, dn, dw, db, dout, N, Cin, Cout, V, 0);
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int iters = 30; cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch_subm_conv_cuda(df, dn, dw, db, dout, N, Cin, Cout, V, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float t; cudaEventElapsedTime(&t, a, b); ms = t / iters;
    std::vector<float> out((size_t)N*Cout);
    cudaMemcpy(out.data(), dout, (size_t)N*Cout*4, cudaMemcpyDeviceToHost);
    cudaFree(df); cudaFree(dn); cudaFree(dw); cudaFree(db); cudaFree(dout);
    return out;
}
#endif

// crude lookup of flex_gemm baseline min_ms for shape (Ci,Co,N) from timing json
static double baseline_min_ms(const std::string& root, int Cin, int Cout, int N) {
    std::ifstream f(root + "/flexgemm_timing.json");
    if (!f) return -1;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::string key = "subm_K333_Ci" + std::to_string(Cin) + "_Co" + std::to_string(Cout) +
                      "_N" + std::to_string(N);
    auto p = s.find('"' + key + '"');
    if (p == std::string::npos) return -1;
    p = s.find("\"min_ms\"", p); if (p == std::string::npos) return -1;
    p = s.find(':', p) + 1;
    return std::atof(s.c_str() + p);
}

// reuse the validated rulebook builder
static int64_t ckey(int32_t b, int32_t z, int32_t y, int32_t x) {
    auto m = [](int32_t v) { return (int64_t)(v & 0xFFFFF); };
    return (((((int64_t)b << 20) | m(z)) << 20 | m(y)) << 20) | m(x);
}
#include <unordered_map>
static std::vector<uint32_t> build_nmap(const int32_t* c, int N, int K) {
    int cc = (K - 1) / 2, V = K * K * K;
    std::unordered_map<int64_t, int> idx; idx.reserve(N * 2);
    for (int i = 0; i < N; i++) idx[ckey(c[i*4], c[i*4+1], c[i*4+2], c[i*4+3])] = i;
    std::vector<uint32_t> nm((size_t)N * V, 0xFFFFFFFFu);
    for (int i = 0; i < N; i++) {
        int32_t b=c[i*4], z=c[i*4+1], y=c[i*4+2], x=c[i*4+3];
        int v = 0;
        for (int kz=0;kz<K;kz++)for(int ky=0;ky<K;ky++)for(int kx=0;kx<K;kx++,v++){
            auto it=idx.find(ckey(b,z+kz-cc,y+ky-cc,x+kx-cc));
            if(it!=idx.end()) nm[(size_t)i*V+v]=it->second;
        }
    }
    return nm;
}

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : "golden_model";
    DIR* d = opendir(root.c_str());
    if (!d) { printf("cannot open %s\n", root.c_str()); return 2; }
    std::vector<std::string> cases;
    for (dirent* e; (e = readdir(d));) {
        std::string n = e->d_name;
        if (n[0] == '.') continue;
        if (std::ifstream(root + "/" + n + "/out_feats.npy")) cases.push_back(n);
    }
    closedir(d);
    std::sort(cases.begin(), cases.end());
    printf("test_subm (fp32 kernel path): %zu cases\n", cases.size());
    int fails = 0;
    for (auto& c : cases) {
        std::string dir = root + "/" + c;
        auto coords = npy_load(dir + "/in_coords.npy");
        auto feats  = npy_load(dir + "/in_feats.npy");
        auto weight = npy_load(dir + "/weight.npy");
        auto bias   = npy_load(dir + "/bias.npy");
        auto gold   = npy_load(dir + "/out_feats.npy");
        int N = (int)coords.shape[0], Cin = (int)feats.shape[1], Cout = (int)weight.shape[2];
        int V = (int)weight.shape[0], K = (V==27?3:(V==1?1:(int)(std::cbrt((double)V)+0.5)));
        if ((int)gold.shape[0] != N) {  // submanifold only: out N must == in N
            printf("  %-28s SKIP (non-submanifold, out N=%lld != %d)\n",
                   c.c_str(), (long long)gold.shape[0], N);
            continue;
        }
        auto nm = build_nmap(coords.i32(), N, K);
        std::vector<float> out; double ms = -1;
#ifdef USE_CUDA
        out = run_cuda(feats.f32(), nm.data(), weight.f32(), bias.f32(), N, Cin, Cout, V, ms);
#else
        out = launch_subm_conv_cpu(feats.f32(), nm.data(), weight.f32(), bias.f32(), N, Cin, Cout, V);
#endif
        double maxabs = 0, maxrel = 0, gmax = 0;
        for (size_t i = 0; i < out.size(); i++) {
            double e = std::fabs(out[i] - gold.f32()[i]);
            maxabs = std::max(maxabs, e);
            gmax = std::max(gmax, (double)std::fabs(gold.f32()[i]));
        }
        maxrel = gmax > 0 ? maxabs / gmax : maxabs;
        bool pass = maxrel < 1e-4;   // fp32-accumulate tolerance vs f64 golden
        char perf[80] = "";
        if (ms >= 0) {
            double base = baseline_min_ms(root, Cin, Cout, N);
            if (base > 0) snprintf(perf, 80, "  ours=%.3fms flexgemm=%.3fms (%.2fx)", ms, base, base / ms);
            else          snprintf(perf, 80, "  ours=%.3fms", ms);
        }
        printf("  %-28s N=%-7d Ci=%-4d Co=%-4d  maxrel=%.2e%s  %s\n",
               c.c_str(), N, Cin, Cout, maxrel, perf, pass ? "PASS" : "*** FAIL ***");
        fails += !pass;
    }
    printf("%s (%d/%zu)\n", fails ? "FAILURES" : "ALL PASS",
           (int)cases.size() - fails, cases.size());
    return fails ? 1 : 0;
}
