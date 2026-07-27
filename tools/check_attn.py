"""
Reproduce the self-attention (cond/noise split, num_cond_latents=1) on the C++
post-rope q/k/v, two ways, to localize whether the cond-split is the bug:
  (REF)  reference avatar/attention.py: q_cond x {k,v}_cond ; q_noise x {k,v}_full
  (PLAIN) full attention, no split
Compare both reproductions against each other; also reproduce exactly what the
C++ self_attn does (same as REF in intent) to verify the C++ split logic.
All pre-proj. We compare REF-split vs PLAIN to see how different the split makes
the output, and print per-region (cond rows vs noise rows) so we see which
region the C++ gets wrong.
"""
import os, struct
import numpy as np

D = "models/_dump"


def rd(f):
    with open(os.path.join(D, f), "rb") as fh:
        nd = struct.unpack("<q", fh.read(8))[0]
        dims = [struct.unpack("<q", fh.read(8))[0] for _ in range(nd)]
        data = np.frombuffer(fh.read(), dtype="<f4").copy()
    return data, dims


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def attn(q, k, v):
    # q,k,v: [head, Lq/Lk, d]
    d = q.shape[-1]
    s = (q @ k.transpose(0, 2, 1)) / np.sqrt(d)  # [head, Lq, Lk]
    return softmax(s) @ v  # [head, Lq, d]


def main():
    H, n_token, d = 32, 3120, 128
    ncl_thw = 1560  # num_cond_latents=1 * (N//N_t)=1560

    q, _ = rd("sa_q_postrope.bin")  # ggml [d, token, head] -> np (head, token, d)
    q = q.reshape(H, n_token, d)
    k, _ = rd("sa_k_postrope.bin")
    k = k.reshape(H, n_token, d)
    v, vd = rd("sa_v.bin")          # ggml [d, head, token, N] -> np (N, token, head, d)
    v = v.reshape(1, n_token, H, d)[0].transpose(1, 0, 2)  # (head, token, d)
    print("shapes q,k,v:", q.shape, k.shape, v.shape, "v ggml dims", vd)

    # REF split
    x_cond = attn(q[:, :ncl_thw], k[:, :ncl_thw], v[:, :ncl_thw])     # [H, ncl, d]
    x_noise = attn(q[:, ncl_thw:], k, v)                             # [H, Lnoise, d]
    ref_split = np.concatenate([x_cond, x_noise], axis=1)            # [H, n_token, d]

    # PLAIN
    plain = attn(q, k, v)

    def cos(a, b):
        a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)

    print("cos(ref_split, plain) overall:", cos(ref_split, plain))
    print("  cond-region  cos:", cos(ref_split[:, :ncl_thw], plain[:, :ncl_thw]))
    print("  noise-region cos:", cos(ref_split[:, ncl_thw:], plain[:, ncl_thw:]))

    # Compare against C++ attn_out (post-proj). We can't apply proj here, but we
    # CAN check the C++ pre-proj by undoing proj is hard; instead, reshape the
    # C++ b0_attn_out and compare its CORRELATION STRUCTURE is not trivial.
    # Key question answered above: how much does the cond split change rows.

    # Reproduce EXACTLY the C++ flatten of the attention output -> [C, n_token]
    # to feed proj externally if needed. C++ out is [d, n_token, head]? The
    # attention returns [N, L_q, C] with C = d*head, head-major or d-major?
    # ggml_ext_attention_ext returns reshape_3d(kqv, d_head*n_head, L_q, N):
    # element = d + d_head*head -> per token vector is [head0_d0..head0_d127, head1_d0..].
    # Build ref_split in that layout: (token, head, d) -> flatten (head,d)
    ref_split_tok = ref_split.transpose(1, 0, 2).reshape(n_token, H * d)  # (token, head*d)
    plain_tok = plain.transpose(1, 0, 2).reshape(n_token, H * d)
    print("ref_split flat std:", ref_split_tok.std(), "plain flat std:", plain_tok.std())


if __name__ == "__main__":
    main()
