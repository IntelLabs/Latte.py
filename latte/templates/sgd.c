#include <omp.h>

extern "C"
void sgd_update(float *param, float *grad, float *hist, float lr, float mom, int length) {
#pragma omp parallel for
    for (int i = 0; i < length; i++) {
        float tmp = (hist[i] * mom) + (grad[i] * -lr);
        param[i] += tmp;
        hist[i] = tmp;
    }
}
