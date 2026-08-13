// qwen_fa2_contract_test.cpp -- graph-level guard for the operation-local
// causal Qwen GQA FlashAttention contract. No backend or model weights needed.
#include <ggml.h>
#include <cstdio>
#include <cstdint>

static bool expect(bool value, const char * what) {
    if (value) return true;
    std::fprintf(stderr, "FAIL: %s\n", what);
    return false;
}

static int32_t op_i32(const ggml_tensor * t, int index) {
    return t->op_params[index];
}

int main() {
    ggml_init_params ip{ 1u << 20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, 514, 16, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, 514,  8, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, 514,  8, 1);

    // Null-mask generic attention is not causality: it must retain normal
    // F32 output even when its dimensions happen to match Qwen GQA.
    ggml_tensor * generic = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 0.1f, 0.0f, 0.0f);
    bool ok = expect(generic->type == GGML_TYPE_F32, "generic null-mask Qwen-shaped attention stays F32") &&
              expect(op_i32(generic, 4) == 0, "generic attention has no causal-Qwen flag");

    ggml_tensor * causal = ggml_flash_attn_ext_qwen_causal_gqa(ctx, q, k, v, 0.1f);
    ok = expect(causal->type == GGML_TYPE_BF16, "Qwen causal GQA contract uses BF16 FA2 output") && ok;
    ok = expect(op_i32(causal, 4) == 1, "Qwen causal GQA flag is set") && ok;

    // The dedicated API is still shape-gated in ggml; it cannot turn an
    // arbitrary GQA-like tensor into the FA2 causal operation.
    ggml_tensor * bad_k = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, 514, 16, 1);
    ggml_tensor * bad = ggml_flash_attn_ext_qwen_causal_gqa(ctx, q, bad_k, bad_k, 0.1f);
    ok = expect(bad->type == GGML_TYPE_F32, "wrong Qwen GQA shape falls back to generic F32") && ok;
    ok = expect(op_i32(bad, 4) == 0, "wrong Qwen GQA shape has no causal flag") && ok;

    ggml_free(ctx);
    if (ok) std::puts("qwen FA2 operation-local contract: PASS");
    return ok ? 0 : 1;
}
