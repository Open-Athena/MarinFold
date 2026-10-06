// Uniform-stationary bipartite edge-switch chain for the exp350 contact null.
// Every proposal, including a rejected move, counts toward burn-in/thinning.
// The proposal is symmetric: choose two edge slots uniformly and exchange
// their right endpoints if the new edges are distinct and absent. Left slots
// never change; right endpoints are only permuted, preserving both degrees.
// Finite-time draws are approximate; the analysis compares longer chains.

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

extern "C" void sample_degree_null(
    const int32_t* left, const int32_t* right, const int64_t* offsets,
    int n_maps, int n_left, int n_right, const uint8_t* truth,
    int n_draws, int burn_per_edge, int thin_per_edge, uint64_t seed,
    int n_threads, int32_t* true_positives, double* acceptance,
    double* overlap, int32_t* final_right
) {
    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (int sample = 0; sample < n_maps; ++sample) {
        const int64_t start = offsets[sample];
        const int p = static_cast<int>(offsets[sample + 1] - start);
        if (p == 0) continue;
        std::vector<int32_t> y(right + start, right + start + p);
        std::vector<uint8_t> adjacency(n_left * n_right, 0);
        std::vector<uint8_t> original(n_left * n_right, 0);
        for (int e = 0; e < p; ++e) {
            const int index = left[start + e] * n_right + y[e];
            adjacency[index] = original[index] = 1;
        }
        std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * (sample + 1));
        std::uniform_int_distribution<int> edge(0, p - 1);
        int64_t accepted = 0;
        int64_t proposed = 0;
        double overlap_sum = 0;
        for (int draw = -1; draw < n_draws; ++draw) {
            const int steps = p * (draw == -1 ? burn_per_edge : thin_per_edge);
            for (int step = 0; step < steps; ++step) {
                ++proposed;
                const int a = edge(rng), b = edge(rng);
                const int u = left[start + a], v = left[start + b];
                const int x = y[a], z = y[b];
                if (u == v || x == z) continue;
                if (adjacency[u * n_right + z] || adjacency[v * n_right + x]) continue;
                adjacency[u * n_right + x] = adjacency[v * n_right + z] = 0;
                adjacency[u * n_right + z] = adjacency[v * n_right + x] = 1;
                std::swap(y[a], y[b]);
                ++accepted;
            }
            if (draw == -1) continue;
            int tp = 0, retained = 0;
            for (int e = 0; e < p; ++e) {
                const int index = left[start + e] * n_right + y[e];
                tp += truth[index];
                retained += original[index];
            }
            true_positives[draw * n_maps + sample] = tp;
            overlap_sum += static_cast<double>(retained) / p;
        }
        acceptance[sample] = proposed ? static_cast<double>(accepted) / proposed : 0.0;
        overlap[sample] = overlap_sum / n_draws;
        std::copy(y.begin(), y.end(), final_right + start);
    }
}
