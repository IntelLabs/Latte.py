/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <omp.h>

#define SIMDWITH 8

extern "C"
void reorder_4d_5d(float *_input, float *_output, int dim0, int dim1, int dim2, int dim3) {
    float (* __restrict input)[dim1][dim2][dim3] = (float (*)[dim1][dim2][dim3]) _input;
    float (* __restrict output)[dim1 / SIMDWITH][dim2][dim3][SIMDWITH] = (float (*)[dim1 / SIMDWITH][dim2][dim3][SIMDWITH]) _output;
    //float (* input)[dim1][dim2][dim3] = (float (*)[dim1][dim2][dim3]) _input;
    //float (* output)[dim1 / SIMDWITH][dim2][dim3][SIMDWITH] = (float (*)[dim1 / SIMDWITH][dim2][dim3][SIMDWITH]) _output;
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
