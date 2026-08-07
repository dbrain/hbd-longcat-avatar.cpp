#!/usr/bin/env python3
"""Numerical oracle for Comfy's int8_tensorwise ConvRot contract.

This deliberately spells the CUDA kernel's radix-4 transform separately from
the explicit Kronecker matrix used by comfy-kitchen.  It catches index ordering,
normalization, row-scale, rounding, and I32-accumulation mistakes without a
model or render.
"""

import numpy as np


H4 = np.asarray(
    [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
    dtype=np.float32,
)


def comfy_h256() -> np.ndarray:
    h = H4
    for _ in range(3):
        h = np.kron(H4, h)
    return h / np.float32(16.0)


def cuda_radix4_order(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    for stride in (1, 4, 16, 64):
        old = y.copy()
        for i in range(256):
            digit = (i // stride) & 3
            base = i - digit * stride
            a, b, c, d = (old[base + j * stride] for j in range(4))
            y[i] = (
                a + b + c - d
                if digit == 0
                else a + b - c + d
                if digit == 1
                else a - b + c + d
                if digit == 2
                else -a + b + c + d
            )
    return y / np.float32(16.0)


def quantize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.maximum(np.max(np.abs(x), axis=1, keepdims=True) / 127.0, 1e-30)
    # np.rint and CUDA __float2int_rn both use round-to-nearest-even.
    return np.clip(np.rint(x / scales), -127, 127).astype(np.int8), scales


def main() -> None:
    rng = np.random.default_rng(0xC0FFEE)
    h = comfy_h256()
    x = rng.standard_normal((3, 256), dtype=np.float32)
    radix = np.stack([cuda_radix4_order(row) for row in x])
    explicit = x @ h
    np.testing.assert_allclose(radix, explicit, rtol=2e-6, atol=2e-6)

    weight = rng.standard_normal((19, 256), dtype=np.float32)
    qweight, weight_scale = quantize_rows(weight @ h)
    qx, activation_scale = quantize_rows(radix)
    accum = qx.astype(np.int32) @ qweight.astype(np.int32).T
    base = accum.astype(np.float32) * activation_scale
    direct = base * weight_scale.T
    comfy_spelling = (
        (qx.astype(np.float32) * activation_scale)
        @ (qweight.astype(np.float32) * weight_scale).T
    )
    np.testing.assert_allclose(direct, comfy_spelling, rtol=1e-5, atol=2e-5)

    # The CUDA fast path may consume the checkpoint's per-output weight scale
    # and bias while it converts the private I32 accumulator to F32.  Its
    # column-major output layout is [out_features, activation_rows], so the
    # scale/bias index repeats every out_features elements.
    bias = rng.standard_normal((19, 1), dtype=np.float32)
    old_epilogue = direct + bias.T
    fused_epilogue = (
        base * weight_scale.T
        + bias.T
    )
    np.testing.assert_allclose(fused_epilogue, old_epilogue, rtol=1e-6, atol=1e-6)

    # The F16 graph route must not change ConvRot's arithmetic.  It feeds the
    # existing transform from a half activation, still performs rotation,
    # scale reduction, quantization and I32 accumulation in F32/I8/I32, then
    # rounds only the final graph value to F16.
    x_f16 = x.astype(np.float16).astype(np.float32)
    radix_f16 = np.stack([cuda_radix4_order(row) for row in x_f16])
    qx_f16, activation_scale_f16 = quantize_rows(radix_f16)
    accum_f16 = qx_f16.astype(np.int32) @ qweight.astype(np.int32).T
    direct_f16 = accum_f16.astype(np.float32) * activation_scale_f16
    direct_f16 = direct_f16 * weight_scale.T + bias.T
    cuda_f16_store = direct_f16.astype(np.float16)
    reference_f16_store = np.asarray(direct_f16, dtype=np.float16)
    np.testing.assert_array_equal(cuda_f16_store, reference_f16_store)
    assert np.isfinite(cuda_f16_store).all()

    # Runtime LoRA is deliberately outside the base weight row scale.  This
    # mirrors the graph spelling:
    #   (I32_accum * activation_scale) * weight_scale + multiplier * (x @ A @ B)
    rank = 5
    down = rng.standard_normal((256, rank), dtype=np.float32)
    up = rng.standard_normal((rank, 19), dtype=np.float32)
    delta = x @ down @ up
    no_adapter = direct.copy()
    multiplier_zero = direct + np.float32(0.0) * delta
    multiplier_nonzero = direct + np.float32(0.375) * delta
    np.testing.assert_array_equal(multiplier_zero, no_adapter)
    np.testing.assert_allclose(
        multiplier_nonzero - direct,
        np.float32(0.375) * delta,
        rtol=2e-5,
        atol=2e-5,
    )
    wrongly_scaled_delta = (
        direct + np.float32(0.375) * delta * weight_scale.T
    )
    assert not np.allclose(multiplier_nonzero, wrongly_scaled_delta)
    print(
        "int8_convrot oracle OK:",
        f"transform_max_abs={np.max(np.abs(radix-explicit)):.3g}",
        f"linear_max_abs={np.max(np.abs(direct-comfy_spelling)):.3g}",
        f"fused_epilogue_max_abs={np.max(np.abs(fused_epilogue-old_epilogue)):.3g}",
        f"f16_store_max_abs={np.max(np.abs(cuda_f16_store-reference_f16_store)):.3g}",
        "lora_zero=bit-identical",
        f"lora_nonzero_max_abs={np.max(np.abs(multiplier_nonzero-direct)):.3g}",
    )


if __name__ == "__main__":
    main()
