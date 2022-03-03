#include <benchmark/benchmark.h>
#include <stdlib.h>

//#define VALIDATION

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

#ifdef __ARM_NEON
#include <arm_neon.h>

int sum_neon_4x(int *arr, int N) {
    constexpr int M = 4; // unroll factor
    int res = 0;

    int i;
    no_auto_vectorize for (i = 0; i < N; i += M, arr += M) {
        int32x4_t four_ints = vld1q_s32(arr);
        int32_t acc = vaddvq_s32(four_ints);
        res += acc;
    }

    no_auto_vectorize for (i -= M; i < N; i++) { res += arr[i]; }

    return res;
}
#endif

#define SUM_UNROLL_IMPL(UNROLL_FACTOR)                                         \
    int sum_unroll_##UNROLL_FACTOR##x(int *arr, int N) {                       \
        int res = 0;                                                           \
                                                                               \
        constexpr int M = UNROLL_FACTOR;                                       \
                                                                               \
        int i;                                                                 \
        int lim = N - N % M;                                                   \
        no_auto_vectorize for (i = 0; i < lim; i += M) {                       \
            for (int j = 0; j < M; j++) {                                      \
                res += arr[i + j];                                             \
            }                                                                  \
        }                                                                      \
                                                                               \
        for (; i < N; i++) {                                                   \
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

#ifdef __ARM_NEON
BENCH_SUM(neon_4x);
#endif

BENCH_SUM(unroll_4x);
//BENCH_SUM(unroll_8x);
//BENCH_SUM(unroll_16x);
//BENCH_SUM(unroll_32x);
BENCH_SUM(naive);
//BENCH_SUM(auto_vec);

#ifdef VALIDATION
#include <assert.h>
#define assert_eq(LHS, RHS)                                                    \
    do {                                                                       \
        if ((LHS) != (RHS)) {                                                  \
            fprintf(stderr, "lhs: %d != rhs: %d\n", LHS, RHS);                 \
            assert((LHS) == (RHS) && "assert_eq failed.");                     \
        }                                                                      \
    } while (0)

int main() {
    constexpr int N = 1000;
    int *arr = prepare_array(N);
    int ground_truth = sum_naive(arr, N);

    assert_eq(ground_truth, sum_neon_4x(arr, N));
    assert_eq(ground_truth, sum_unroll_4x(arr, N));
    assert_eq(ground_truth, sum_unroll_8x(arr, N));
    assert_eq(ground_truth, sum_unroll_16x(arr, N));
    assert_eq(ground_truth, sum_auto_vec(arr, N));

    printf("Tests passed!\n");
}
#else
BENCHMARK_MAIN();
#endif
