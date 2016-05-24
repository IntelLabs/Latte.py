#include <omp.h>

extern "C"
void sgd_update(float *__restrict param, float *__restrict grad, float
        *__restrict hist, float lr, float mom, int length, int _num_threads) {
#pragma omp parallel for simd schedule(static, 8)
    for (int i = 0; i < length; i++) {
        float _grad = 0.0;
        for (long j = 0; j < _num_threads; j++) {
            _grad += grad[j * length + i];
        }
        float tmp = (hist[i] * mom) + (_grad * lr);
        param[i] -= tmp;
        hist[i] = tmp;
    }
}
