#include <omp.h>

extern "C"
void sgd_update(float *__restrict param, float *__restrict grad, float
        *__restrict hist, float lr, float mom, int length) {
#pragma omp parallel for simd
    for (int i = 0; i < length; i++) {
        float tmp = (hist[i] * mom) + (grad[i] * -lr);
        param[i] += tmp;
        hist[i] = tmp;
    }
}
