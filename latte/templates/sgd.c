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
#include <immintrin.h>
#define SIMDWIDTH 8

extern "C"
void sgd_update(float *__restrict param, float *__restrict grad, float
        *__restrict hist, float lr, float mom, int length, int _num_threads) {
    __assume_aligned(grad, 64);
    __assume_aligned(hist, 64);
    __assume_aligned(param, 64);
    __assume(length%16==0);
    #pragma omp parallel for simd
    for (int i = 0; i < length; i++) {
        float _grad = 0.0;
        // for (long j = 0; j < _num_threads; j++) {
        //     _mm_prefetch((char*)(grad + (j + 1) * length + i),_MM_HINT_T0);
        //     _grad += grad[j * length + i];
        // }
        //float tmp = (hist[i] * mom) + (_grad * lr);
        float tmp = (hist[i] * mom) + (grad[i] * lr);
        param[i] -= tmp;
        hist[i] = tmp;
    }
}
