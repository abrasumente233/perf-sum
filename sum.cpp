#include <benchmark/benchmark.h>
#include <stdlib.h>

static int *prepare_array(int N) {
    int *arr = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        arr[i] = i+1;
    }

    return arr;
}

int sum_naive(int *arr, int N) {
    int res = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (int i = 0; i < N; i++) {
        res += arr[i];
    }
    return res;
}

int sum_auto_vec(int *arr, int N) {
    int res = 0;
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < N; i++) {
        res += arr[i];
    }
    return res;
}

#define BENCH_SUM(NAME)                                                        \
    static void bench_##NAME(benchmark::State &state) {                        \
        int N = state.range(0);                                                \
        int *arr = prepare_array(N);                                           \
                                                                               \
        for (auto _ : state) {                                                 \
            benchmark::DoNotOptimize(sum_##NAME(arr, N));                      \
        }                                                                      \
                                                                               \
        state.SetItemsProcessed(N);                                            \
                                                                               \
        free(arr);                                                             \
    }                                                                          \
    BENCHMARK(bench_##NAME)->Arg(10)->Arg(1000)->Arg(10000)->Arg(1000000)

BENCH_SUM(naive);
BENCH_SUM(auto_vec);

BENCHMARK_MAIN();
