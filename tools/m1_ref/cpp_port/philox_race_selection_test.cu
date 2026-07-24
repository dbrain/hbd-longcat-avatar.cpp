// End-to-end selection fixture for the sampled-beam primitive.  It deliberately
// does not include or link the decoder: fixed [10,16] logits are warped using
// the same temperature -> top-k(min=2) -> top-p(min=2) ordering as
// rig_beam_generate_batched.hpp, then the native PyTorch Philox exponential
// stream selects 2*num_beams entries from the flattened [B,V] distribution.
//
// Expected picks were captured on the assigned RTX 3060 immediately before
// this test with torch 2.7.0+cu128:
//   torch.multinomial(torch.softmax(warped, -1).reshape(1,-1), 20,
//                     replacement=False, generator=Generator("cuda").manual_seed(0))
// Build: ./build.sh philox_race_selection_test cuda
#include "rig_philox_race.hpp"

#include <cuda_runtime.h>
#include <cuda.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int B = 10, V = 16, WANT = 20;
constexpr std::array<unsigned char, 16> k3060Uuid = {
    0x3b, 0x9a, 0xc5, 0xcf, 0x95, 0xc5, 0x5c, 0x9e,
    0xde, 0x19, 0xaf, 0x33, 0xaf, 0x4b, 0x27, 0xd6,
};

void require_3060() {
    CUuuid uuid{};
    cudaDeviceProp prop{};
    CUdevice device{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS || cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS)
        throw std::runtime_error("cannot query selected CUDA device");
    if (std::memcmp(uuid.bytes, k3060Uuid.data(), k3060Uuid.size()) != 0)
        throw std::runtime_error("refusing to run: CUDA device 0 is not RTX 3060 UUID GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6");
    if (prop.major != 8 || prop.minor != 6 || std::strstr(prop.name, "RTX 3060") == nullptr)
        throw std::runtime_error("selected UUID does not identify the expected RTX 3060");
}

std::vector<float> fixed_logits() {
    std::vector<float> x(B * V);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < V; ++i)
            x[b * V + i] = ((i * 13 + b * 7) % 31 - 15) * 0.17f + (i == (3 * b + 1) % V ? 0.41f : 0.0f);
    x[2 * V + 14] = -std::numeric_limits<float>::infinity();
    x[7 * V + 2] = -std::numeric_limits<float>::infinity();
    return x;
}

// Local, test-only scalar spelling of the production warper contract.  Values
// intentionally have no threshold ties; a future tie-policy test belongs at
// the logit-warper level, not in the Philox selection test.
void warp_row(float* x) {
    for (int i = 0; i < V; ++i) if (std::isfinite(x[i])) x[i] /= 0.7f;
    std::array<std::pair<float, int>, V> order{};
    int n = 0;
    for (int i = 0; i < V; ++i) if (std::isfinite(x[i])) order[n++] = {x[i], i};
    std::sort(order.begin(), order.begin() + n, [](auto a, auto b) { return a.first > b.first; });
    const float kth = order[4].first;
    for (int i = 0; i < V; ++i) if (x[i] < kth) x[i] = -std::numeric_limits<float>::infinity();

    n = 0;
    for (int i = 0; i < V; ++i) if (std::isfinite(x[i])) order[n++] = {x[i], i};
    std::sort(order.begin(), order.begin() + n, [](auto a, auto b) { return a.first > b.first; });
    const float mx = order[0].first;
    float sum = 0.0f;
    for (int r = 0; r < n; ++r) sum += std::exp(order[r].first - mx);
    float cum = 0.0f;
    for (int r = 0; r < n; ++r) {
        cum += std::exp(order[r].first - mx) / sum;
        if (cum >= 0.88f && r + 1 >= 2) {
            for (int q = r + 1; q < n; ++q) x[order[q].second] = -std::numeric_limits<float>::infinity();
            break;
        }
    }
}

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> p(x.size());
    for (int b = 0; b < B; ++b) {
        const float* row = x.data() + b * V;
        float mx = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < V; ++i) mx = std::max(mx, row[i]);
        float sum = 0.0f;
        for (int i = 0; i < V; ++i) if (std::isfinite(row[i])) sum += std::exp(row[i] - mx);
        for (int i = 0; i < V; ++i) p[b * V + i] = std::isfinite(row[i]) ? std::exp(row[i] - mx) / sum : 0.0f;
    }
    return p;
}

std::vector<int> exponential_race(const std::vector<float>& p) {
    std::vector<float> exponential(p.size());
    uint64_t offset = 0;
    if (!rig_torch_philox_exponentials(exponential, /*seed=*/0, &offset) || offset != 4)
        throw std::runtime_error("native Philox call failed or consumed unexpected offset");
    struct Race { float key; int index; };
    std::vector<Race> race; race.reserve(p.size());
    for (int i = 0; i < (int)p.size(); ++i)
        race.push_back({p[i] > 0.0f ? exponential[i] / p[i] : std::numeric_limits<float>::infinity(), i});
    std::partial_sort(race.begin(), race.begin() + WANT, race.end(),
                      [](const Race& a, const Race& b) { return a.key < b.key; });
    std::vector<int> picks; picks.reserve(WANT);
    for (int r = 0; r < WANT; ++r) picks.push_back(race[r].index);
    return picks;
}

void check_zero_probability_tail() {
    // torch 2.7 CUDA `multinomial([.6,.4,0,0,0,0], 6, replacement=False)`
    // returns both positive-probability elements before its arbitrary zero tail
    // (its observed zero order is [4,5,2,3], which is not an API guarantee).
    // The native key is +inf for p==0, so make the portable invariant explicit:
    // finite keys first, then each zero-probability index once.  Sampled-beam's
    // top-p minimum guarantees >= 2 candidates per beam, hence the production
    // 10-beam / want=20 call never needs this tail.
    std::vector<float> p = {0.6f, 0.4f, 0, 0, 0, 0};
    std::vector<float> e(p.size()); uint64_t offset = 0;
    if (!rig_torch_philox_exponentials(e, 0, &offset)) throw std::runtime_error("zero-tail Philox call failed");
    struct Key { float key; int index; };
    std::vector<Key> keys;
    for (int i = 0; i < 6; ++i) keys.push_back({p[i] ? e[i] / p[i] : std::numeric_limits<float>::infinity(), i});
    std::partial_sort(keys.begin(), keys.end(), keys.end(), [](const Key& a, const Key& b) { return a.key < b.key; });
    if (keys[0].index > 1 || keys[1].index > 1) throw std::runtime_error("zero-probability item preceded finite race key");
    bool seen[6]{};
    for (const auto& x : keys) {
        if (seen[x.index]) throw std::runtime_error("zero-tail race produced duplicate index");
        seen[x.index] = true;
    }
}
} // namespace

int main() try {
    require_3060();
    auto logits = fixed_logits();
    for (int b = 0; b < B; ++b) warp_row(logits.data() + b * V);
    const auto picks = exponential_race(softmax(logits));
    constexpr std::array<int, WANT> expected = {
        146, 151, 127, 20, 140, 33, 7, 58, 51, 64,
        107, 9, 76, 102, 158, 63, 40, 84, 133, 22,
    };
    if (!std::equal(picks.begin(), picks.end(), expected.begin())) {
        std::fprintf(stderr, "FAIL: sampled flattened indices native:");
        for (int x : picks) std::fprintf(stderr, " %d", x);
        std::fprintf(stderr, "\ntorch:");
        for (int x : expected) std::fprintf(stderr, " %d", x);
        std::fprintf(stderr, "\nnative survivors:");
        for (int b = 0; b < B; ++b) {
            std::fprintf(stderr, " [");
            for (int i = 0; i < V; ++i) if (std::isfinite(logits[b * V + i])) std::fprintf(stderr, "%d,", i);
            std::fprintf(stderr, "]");
        }
        std::fprintf(stderr, "\n");
        return 1;
    }
    check_zero_probability_tail();
    std::printf("PASS: torch 2.7 CUDA flattened [10,16] Philox exponential-race selection matches 20/20 picks on RTX 3060 UUID; mapping: key=exponential_(1)[flat] / softmax(warped)[flat], ascending key; zero-probability +inf tail stays after finite keys\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 2;
}
