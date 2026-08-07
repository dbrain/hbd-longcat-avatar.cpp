#!/usr/bin/env python3
"""Small deterministic CPU oracle for the native H3 Sol-Attn algebra.

This deliberately mirrors the CUDA adapter's native-layout-independent math:
64-token K means, V sums, Gaussian block routing, adjacent exact blocks, and
the optional exact conditioning prefix.  It is not a performance benchmark.
"""

import math
import random


BLOCK = 64


def dense(q, k, v, scale):
    scores = [sum(a * b for a, b in zip(q, key)) * scale for key in k]
    top = max(scores)
    weights = [math.exp(score - top) for score in scores]
    total = sum(weights)
    return [sum(weight * value[d] for weight, value in zip(weights, v)) / total
            for d in range(len(q))]


def sol(qs, ks, vs, scale, tau, sink_tokens=0, sink_query_tokens=0,
        force_exact=False):
    tokens, dim = len(qs), len(qs[0])
    blocks = (tokens + BLOCK - 1) // BLOCK
    kc, vc = [], []
    for kb in range(blocks):
        begin, end = kb * BLOCK, min(tokens, (kb + 1) * BLOCK)
        n = end - begin
        kc.append([sum(ks[t][d] for t in range(begin, end)) / n for d in range(dim)])
        vc.append([sum(vs[t][d] for t in range(begin, end)) for d in range(dim)])
    thresholds = []
    for qb in range(blocks):
        begin, end = qb * BLOCK, min(tokens, (qb + 1) * BLOCK)
        centroid = [sum(qs[t][d] for t in range(begin, end)) / (end - begin) for d in range(dim)]
        scores = [sum(a * b for a, b in zip(centroid, key)) * scale for key in kc]
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        thresholds.append(mean + tau * math.sqrt(variance + 1e-6))
    sink_blocks = (sink_tokens + BLOCK - 1) // BLOCK
    sink_q_blocks = (sink_query_tokens + BLOCK - 1) // BLOCK
    output = []
    for t, q in enumerate(qs):
        qb = t // BLOCK
        terms = []
        for kb in range(blocks):
            summary_score = sum(a * b for a, b in zip(q, kc[kb])) * scale
            exact = (force_exact or qb < sink_q_blocks or kb < sink_blocks
                     or abs(qb - kb) <= 1 or summary_score > thresholds[qb])
            if exact:
                terms.extend((sum(a * b for a, b in zip(q, ks[kt])) * scale, vs[kt])
                            for kt in range(kb * BLOCK, min(tokens, (kb + 1) * BLOCK)))
            else:
                terms.append((summary_score, vc[kb]))
        top = max(score for score, _ in terms)
        weights = [math.exp(score - top) for score, _ in terms]
        total = sum(weights)
        output.append([sum(weight * value[d] for weight, (_, value) in zip(weights, terms)) / total
                       for d in range(dim)])
    return output


def main():
    random.seed(20260805)
    tokens, dim = 129, 8
    q = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(tokens)]
    k = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(tokens)]
    v = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(tokens)]
    scale = dim ** -0.5
    # The CUDA ALL_EXACT diagnostic must reduce to ordinary attention without
    # relying on sink handling, including the ragged third block.
    got = sol(q, k, v, scale, tau=4.0, force_exact=True)
    want = [dense(row, k, v, scale) for row in q]
    error = max(abs(a - b) for row_a, row_b in zip(got, want) for a, b in zip(row_a, row_b))
    assert error < 1e-6, error
    # Ceiling-rounded K and Q sink counts must force every ragged block exact,
    # independently of the threshold.  This is also the sparse-persistent
    # kernel's no-proxy-group edge case.
    sink_k = sol(q, k, v, scale, tau=0.0, sink_tokens=tokens)
    sink_q = sol(q, k, v, scale, tau=0.0, sink_query_tokens=tokens)
    sink_k_error = max(abs(a - b) for row_a, row_b in zip(sink_k, want)
                       for a, b in zip(row_a, row_b))
    sink_q_error = max(abs(a - b) for row_a, row_b in zip(sink_q, want)
                       for a, b in zip(row_a, row_b))
    assert sink_k_error < 1e-6, sink_k_error
    assert sink_q_error < 1e-6, sink_q_error
    # A sparse run must remain finite and deterministic.
    sparse_a = sol(q, k, v, scale, tau=1.0, sink_tokens=17)
    sparse_b = sol(q, k, v, scale, tau=1.0, sink_tokens=17)
    assert sparse_a == sparse_b
    assert all(math.isfinite(x) for row in sparse_a for x in row)
    print("solattn CPU oracle: PASS (all-exact max error %.3g, sinks %.3g/%.3g)" %
          (error, sink_k_error, sink_q_error))


if __name__ == "__main__":
    main()
