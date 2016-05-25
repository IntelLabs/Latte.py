#include <omp.h>
#include <immintrin.h>
#define SIMDWIDTH 8

extern "C"
void sgd_update(float *__restrict param, float *__restrict grad, float
        *__restrict hist, float lr, float mom, int length, int batch_size) {
    __assume_aligned(grad, 64);
    __assume_aligned(hist, 64);
    __assume_aligned(param, 64);
    __assume(length%16==0);
#pragma omp parallel for simd
    for (int i = 0; i < length; i++) {
        float _grad = 0.0;
        for (long j = 0; j < batch_size; j++) {
            _mm_prefetch((char*)(grad + (j + 1) * length + i),_MM_HINT_T0);
            _grad += grad[j * length + i];
        }
        float tmp = (hist[i] * mom) + (_grad * lr);
        param[i] -= tmp;
        hist[i] = tmp;
    }
}
