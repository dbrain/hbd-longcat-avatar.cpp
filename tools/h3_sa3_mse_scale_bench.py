#!/usr/bin/env python3
"""CPU oracle for SA3's neighbouring-E4M3 scale search.

This mirrors the three physical groupings in fattn-sa3.cu: Q/K group 16 adjacent
channels, while transposed V groups 16 adjacent tokens. It is intentionally a
small deterministic correctness/cost gate, not a replacement for CUDA timing on
captured H3 activations.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_folded_nvfp4 import LUTm, e4m3_dec_byte, quant_nvfp4_unfolded  # noqa: E402


def unpack_nvfp4(payload, shape):
    rows, width = shape
    blocks = width // 64
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(rows, blocks, 36)
    scale = raw[:, :, :4]
    packed = raw[:, :, 4:]
    code = np.empty((rows, blocks, 64), dtype=np.uint8)
    for part in range(4):
        pair = packed[:, :, part * 8:part * 8 + 8]
        code[:, :, part * 16:part * 16 + 8] = pair & 15
        code[:, :, part * 16 + 8:part * 16 + 16] = pair >> 4
    value = np.where(code & 8, -LUTm[code & 7], LUTm[code & 7])
    effective = np.repeat(e4m3_dec_byte(scale), 16, axis=2)
    return (value * effective).reshape(shape).astype(np.float32)


def quantize(matrix, radius):
    start = time.perf_counter()
    payload, global_scale = quant_nvfp4_unfolded(
        matrix, flat=True, scale_search_radius=radius)
    elapsed = time.perf_counter() - start
    assert global_scale == 1.0
    return unpack_nvfp4(payload, matrix.shape), elapsed


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def rel_l2(reference, candidate):
    return float(np.linalg.norm(reference.astype(np.float64) - candidate) /
                 np.linalg.norm(reference.astype(np.float64)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    if args.length % 128 or args.length % 64:
        parser.error("--length must be divisible by 128")
    if args.queries <= 0 or args.queries > args.length:
        parser.error("--queries must be in 1..length")

    rng = np.random.default_rng(args.seed)
    shape = (args.heads, args.length, 128)
    # Heavy-tailed samples plus channel offsets exercise the exact centering and
    # outlier-controlled scale choice that make amax/6 suboptimal.
    q = (0.55 * rng.standard_t(5, size=shape) +
         0.20 * rng.normal(size=(args.heads, 1, 128))).astype(np.float16)
    k = (0.50 * rng.standard_t(5, size=shape) +
         0.25 * rng.normal(size=(args.heads, 1, 128))).astype(np.float16)
    v = (0.45 * rng.standard_t(4, size=shape)).astype(np.float16)

    q_blocks = q.reshape(args.heads, args.length // 128, 128, 128)
    q_mean = q_blocks.astype(np.float32).mean(axis=2).astype(np.float16)
    q_centered = (q_blocks - q_mean[:, :, None, :]).astype(np.float16).reshape(shape)
    k_mean = k.astype(np.float32).mean(axis=1, keepdims=True)
    k_centered = (k.astype(np.float32) - k_mean).astype(np.float16)

    q_matrix = q_centered.reshape(-1, 128).astype(np.float32)
    k_matrix = k_centered.reshape(-1, 128).astype(np.float32)
    # V's CUDA kernel transposes first, so its groups run along the token axis.
    v_matrix = v.transpose(0, 2, 1).reshape(-1, args.length).astype(np.float32)
    q_mean_rows = np.repeat(q_mean, 128, axis=1).reshape(shape).astype(np.float32)

    reference_score = (
        q_centered[0, :args.queries].astype(np.float32) @
        k_centered[0].astype(np.float32).T +
        q_mean_rows[0, :args.queries] @ k_centered[0].astype(np.float32).T) / np.sqrt(128.0)
    reference_prob = softmax(reference_score)
    reference_out = reference_prob @ v[0].astype(np.float32)

    print(f"synthetic SA3 oracle H={args.heads} L={args.length} D=128 queries={args.queries}")
    print("radius  q_relL2   k_relL2   v_relL2   score_relL2  output_relL2  q/k/v_cpu_s")
    for radius in (0, 1, 2, 4):
        q_hat, q_time = quantize(q_matrix, radius)
        k_hat, k_time = quantize(k_matrix, radius)
        v_hat, v_time = quantize(v_matrix, radius)
        q_hat = q_hat.reshape(shape)
        k_hat = k_hat.reshape(shape)
        v_hat = v_hat.reshape(args.heads, 128, args.length).transpose(0, 2, 1)
        # SA3's delta term uses exact half K and Q means. Only the centred QK
        # product uses both quantized operands.
        score = (
            q_hat[0, :args.queries] @ k_hat[0].T +
            q_mean_rows[0, :args.queries] @ k_centered[0].astype(np.float32).T) / np.sqrt(128.0)
        output = softmax(score) @ v_hat[0]
        print(f"{radius:>6d}  {rel_l2(q_centered, q_hat):.7f}  "
              f"{rel_l2(k_centered, k_hat):.7f}  {rel_l2(v, v_hat):.7f}  "
              f"{rel_l2(reference_score, score):.7f}      "
              f"{rel_l2(reference_out, output):.7f}      "
              f"{q_time:.3f}/{k_time:.3f}/{v_time:.3f}")


if __name__ == "__main__":
    main()
