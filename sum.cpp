#include <benchmark/benchmark.h>
#include <stdlib.h>
#include <arm_neon.h>

#define no_auto_vectorize                                                      \
    _Pragma("clang loop vectorize(disable) interleave(disable)")
#define auto_vectorize                                                         \
    _Pragma("clang loop vectorize(enable) interleave(enable)")

static int *prepare_array(int N) {
    int *arr = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        arr[i] = i+1;
    }
    return arr;
}

int sum_neon_4x(int *arr, int N) {
    constexpr int M = 4; // unroll factor
    int res = 0;

    int i;
    no_auto_vectorize for (i = 0; i < N; i += M, arr += 4) {
        int32x4_t four_ints = vld1q_s32(arr);
        int32_t acc = vaddvq_s32(four_ints);
        res += acc;
    }

    no_auto_vectorize for (i -= M; i < N; i++) { res += arr[i]; }

    return res;
}

#define SUM_UNROLL_IMPL(UNROLL_FACTOR)                                         \
    int sum_unroll_##UNROLL_FACTOR##x(int *arr, int N) {                       \
        int res = 0;                                                           \
                                                                               \
        constexpr int M = UNROLL_FACTOR;                                       \
                                                                               \
        int i;                                                                 \
        no_auto_vectorize for (i = 0; i < N; i += M) {                         \
            for (int j = 0; j < M; j++) {                                      \
                res += arr[i + j];                                             \
            }                                                                  \
        }                                                                      \
                                                                               \
        for (i -= M; i < N; i++) {                                             \
            res += arr[i];                                                     \
        }                                                                      \
                                                                               \
        return res;                                                            \
    }

SUM_UNROLL_IMPL(4)
SUM_UNROLL_IMPL(8)
SUM_UNROLL_IMPL(16)
SUM_UNROLL_IMPL(32)

int sum_naive(int *arr, int N) {
    int res = 0;
    no_auto_vectorize for (int i = 0; i < N; i++) { res += arr[i]; }
    return res;
}

int sum_auto_vec(int *arr, int N) {
    int res = 0;
    auto_vectorize for (int i = 0; i < N; i++) { res += arr[i]; }
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

BENCH_SUM(neon_4x);
BENCH_SUM(unroll_4x);
//BENCH_SUM(unroll_8x);
//BENCH_SUM(unroll_16x);
//BENCH_SUM(unroll_32x);
//BENCH_SUM(naive);
//BENCH_SUM(auto_vec);

BENCHMARK_MAIN();

/*
int main() {
    int *arr = prepare_array(1000);
    int res = sum_neno_4x(arr, 1000);
    printf("%d\n", res);
}
*/
