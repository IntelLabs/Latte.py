#include <omp.h>

#define SIMDWITH 8

extern "C"
void reorder_4d_5d(float *_input, float *_output, int dim0, int dim1, int dim2, int dim3) {
    float (* __restrict input)[dim1][dim2][dim3] = (float (*)[dim1][dim2][dim3]) _input;
    float (* __restrict output)[dim1 / SIMDWITH][dim2][dim3][SIMDWITH] = (float (*)[dim1 / SIMDWITH][dim2][dim3][SIMDWITH]) _output;
    #pragma omp parallel for collapse(2)
    for (int n = 0; n < dim0; n++) {
        for (int i = 0; i < dim1 / SIMDWITH; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    for (int v = 0; v < SIMDWITH; v++) {
                        output[n][i][j][k][v] = input[n][i * SIMDWITH + v][j][k];
                    }
                }
            }
        }
    }
}
