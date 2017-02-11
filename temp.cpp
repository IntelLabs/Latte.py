// <file: forward0.cpp>
#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#if 0
#include "/nfs_home/avenkat/latte/latte/runtime/runtime.h"
#endif
#define SIMDWIDTH 8
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#if 0
#ifdef APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#if 0
#include <libxsmm.h>
#include <libxsmm_dnn.h>
#endif

inline void __mm_prefetch_t0(float const *a) {
  _mm_prefetch((char const *)a, _MM_HINT_T0);
}

inline void __mm_prefetch_t1(float const *a) {
  _mm_prefetch((char const *)a, _MM_HINT_T1);
}

inline double *calloc_doubles(size_t size) {
  return (double *)calloc(size,sizeof(double));
}

template<int in_width, int out_width>
void transpose(float *in, float *out)
{
    __m256i r0, r1, r2, r3, r4, r5, r6, r7;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7;
    if((in_width & 0x7 != 0)  || (out_width & 0x7 != 0)) {printf("Transpose8x8: Invalid in or out width\\n"); return;}

    r0 = _mm256_load_si256((const __m256i *)(in + 0*in_width));
    r1 = _mm256_load_si256((const __m256i *)(in + 1*in_width));
    r2 = _mm256_load_si256((const __m256i *)(in + 2*in_width));
    r3 = _mm256_load_si256((const __m256i *)(in + 3*in_width));
    r4 = _mm256_load_si256((const __m256i *)(in + 4*in_width));
    r5 = _mm256_load_si256((const __m256i *)(in + 5*in_width));
    r6 = _mm256_load_si256((const __m256i *)(in + 6*in_width));
    r7 = _mm256_load_si256((const __m256i *)(in + 7*in_width));

    t0 = _mm256_unpacklo_epi32(r0,r1); 
    t1 = _mm256_unpackhi_epi32(r0,r1); 
    t2 = _mm256_unpacklo_epi32(r2,r3); 
    t3 = _mm256_unpackhi_epi32(r2,r3); 
    t4 = _mm256_unpacklo_epi32(r4,r5); 
    t5 = _mm256_unpackhi_epi32(r4,r5); 
    t6 = _mm256_unpacklo_epi32(r6,r7); 
    t7 = _mm256_unpackhi_epi32(r6,r7); 

    r0 = _mm256_unpacklo_epi64(t0,t2); 
    r1 = _mm256_unpackhi_epi64(t0,t2); 
    r2 = _mm256_unpacklo_epi64(t1,t3); 
    r3 = _mm256_unpackhi_epi64(t1,t3); 
    r4 = _mm256_unpacklo_epi64(t4,t6); 
    r5 = _mm256_unpackhi_epi64(t4,t6); 
    r6 = _mm256_unpacklo_epi64(t5,t7); 
    r7 = _mm256_unpackhi_epi64(t5,t7); 

    t0 = _mm256_permute2f128_si256(r0, r4, 0x20); 
    t1 = _mm256_permute2f128_si256(r1, r5, 0x20); 
    t2 = _mm256_permute2f128_si256(r2, r6, 0x20); 
    t3 = _mm256_permute2f128_si256(r3, r7, 0x20); 
    t4 = _mm256_permute2f128_si256(r0, r4, 0x31); 
    t5 = _mm256_permute2f128_si256(r1, r5, 0x31); 
    t6 = _mm256_permute2f128_si256(r2, r6, 0x31); 
    t7 = _mm256_permute2f128_si256(r3, r7, 0x31); 

    _mm256_store_si256((__m256i *)(out + 0*out_width), t0);
    _mm256_store_si256((__m256i *)(out + 1*out_width), t1);
    _mm256_store_si256((__m256i *)(out + 2*out_width), t2);
    _mm256_store_si256((__m256i *)(out + 3*out_width), t3);
    _mm256_store_si256((__m256i *)(out + 4*out_width), t4);
    _mm256_store_si256((__m256i *)(out + 5*out_width), t5);
    _mm256_store_si256((__m256i *)(out + 6*out_width), t6);
    _mm256_store_si256((__m256i *)(out + 7*out_width), t7);
}

extern "C"
void forward0(float* _ensemble2inputs, float* _ensemble2value, float* _ensemble2weights, float* _ensemble2weights_transposed, float* _ensemble3bias, float* _ensemble3inputs, float* _ensemble3value, double* _ensemble4alpha, double* _ensemble4beta, float* _ensemble4inputs, double* _ensemble4k, long* _ensemble4n, double* _ensemble4sum_value, float* _ensemble4value) {
    float (* ensemble4value)[8][56][56][8] = (float (*)[8][56][56][8]) _ensemble4value;
    __assume_aligned(ensemble4value, 64);
    double (* ensemble4sum_value)[8][56][56][8] = (double (*)[8][56][56][8]) _ensemble4sum_value;
    __assume_aligned(ensemble4sum_value, 64);
    long (* ensemble4n)[8][56][56][8] = (long (*)[8][56][56][8]) _ensemble4n;
    __assume_aligned(ensemble4n, 64);
    double (* ensemble4k)[56][56][8] = (double (*)[56][56][8]) _ensemble4k;
    __assume_aligned(ensemble4k, 64);
    float (* ensemble4inputs)[10][56][56][8] = (float (*)[10][56][56][8]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    double (* ensemble4beta)[8][56][56][8] = (double (*)[8][56][56][8]) _ensemble4beta;
    __assume_aligned(ensemble4beta, 64);
    double (* ensemble4alpha)[8][56][56][8] = (double (*)[8][56][56][8]) _ensemble4alpha;
    __assume_aligned(ensemble4alpha, 64);
    float (* ensemble3value)[10][56][56][8] = (float (*)[10][56][56][8]) _ensemble3value;
    __assume_aligned(ensemble3value, 64);
    float (* ensemble3inputs)[10][56][56][8] = (float (*)[10][56][56][8]) _ensemble3inputs;
    __assume_aligned(ensemble3inputs, 64);
    float (* ensemble3bias)[1][8] = (float (*)[1][8]) _ensemble3bias;
    __assume_aligned(ensemble3bias, 64);
    float (* ensemble2weights_transposed)[1][3][3][8][8] = (float (*)[1][3][3][8][8]) _ensemble2weights_transposed;
    __assume_aligned(ensemble2weights_transposed, 64);
    float (* ensemble2weights)[1][3][3][8][8] = (float (*)[1][3][3][8][8]) _ensemble2weights;
    __assume_aligned(ensemble2weights, 64);
    float (* ensemble2value)[10][56][56][8] = (float (*)[10][56][56][8]) _ensemble2value;
    __assume_aligned(ensemble2value, 64);
    float (* ensemble2inputs)[1][58][58][8] = (float (*)[1][58][58][8]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    double *times = 0;
    times = calloc_doubles(1);
    times[0] -= omp_get_wtime();
    
    #pragma omp parallel for
    for (int x0 = 0; x0 < 8; x0++) {
      for (int x1 = 0; x1 < 1; x1 ++) {
        for (int x2 = 0; x2 < 3; x2 ++) {
            for (int x3 = 0; x3 < 3; x3 ++) {
                transpose<SIMDWIDTH,SIMDWIDTH>(& ensemble2weights[x0][x1][x2][x3][0][0], & ensemble2weights_transposed[x0][x1][x2][x3][0][0]);
            }
        }
    }
    } 
    #pragma omp parallel for
    for (int _neuron_index_0 = 0; _neuron_index_0 < 8; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 8; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 56; _neuron_index_2 += 1) {
                int in_y = _neuron_index_2 * 1;
                int _input_offset_2 = in_y;
                for (int _neuron_index_3 = 0; _neuron_index_3 < 56; _neuron_index_3 += 14) {
                    int in_x_0 = (_neuron_index_3 + 0) * 1;
                    int in_x_1 = (_neuron_index_3 + 1) * 1;
                    int in_x_2 = (_neuron_index_3 + 2) * 1;
                    int in_x_3 = (_neuron_index_3 + 3) * 1;
                    int in_x_4 = (_neuron_index_3 + 4) * 1;
                    int in_x_5 = (_neuron_index_3 + 5) * 1;
                    int in_x_6 = (_neuron_index_3 + 6) * 1;
                    int in_x_7 = (_neuron_index_3 + 7) * 1;
                    int in_x_8 = (_neuron_index_3 + 8) * 1;
                    int in_x_9 = (_neuron_index_3 + 9) * 1;
                    int in_x_10 = (_neuron_index_3 + 10) * 1;
                    int in_x_11 = (_neuron_index_3 + 11) * 1;
                    int in_x_12 = (_neuron_index_3 + 12) * 1;
                    int in_x_13 = (_neuron_index_3 + 13) * 1;
                    int _input_offset_3_0 = in_x_0;
                    int _input_offset_3_1 = in_x_1;
                    int _input_offset_3_2 = in_x_2;
                    int _input_offset_3_3 = in_x_3;
                    int _input_offset_3_4 = in_x_4;
                    int _input_offset_3_5 = in_x_5;
                    int _input_offset_3_6 = in_x_6;
                    int _input_offset_3_7 = in_x_7;
                    int _input_offset_3_8 = in_x_8;
                    int _input_offset_3_9 = in_x_9;
                    int _input_offset_3_10 = in_x_10;
                    int _input_offset_3_11 = in_x_11;
                    int _input_offset_3_12 = in_x_12;
                    int _input_offset_3_13 = in_x_13;
                    __m256 ___x0_0 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 0)][0]);
                    __m256 ___x0_1 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 1)][0]);
                    __m256 ___x0_2 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 2)][0]);
                    __m256 ___x0_3 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 3)][0]);
                    __m256 ___x0_4 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 4)][0]);
                    __m256 ___x0_5 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 5)][0]);
                    __m256 ___x0_6 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 6)][0]);
                    __m256 ___x0_7 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 7)][0]);
                    __m256 ___x0_8 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 8)][0]);
                    __m256 ___x0_9 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 9)][0]);
                    __m256 ___x0_10 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 10)][0]);
                    __m256 ___x0_11 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 11)][0]);
                    __m256 ___x0_12 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 12)][0]);
                    __m256 ___x0_13 = _mm256_load_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 13)][0]);
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            for (int i_inner = 0; i_inner < 8; i_inner += 1) {
                                __m256 ___x1 = _mm256_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][0][j][k][i_inner][0]);
                                __m256 ___x2_0 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][i_inner]);
                                __m256 ___x2_1 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][i_inner]);
                                __m256 ___x2_2 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][i_inner]);
                                __m256 ___x2_3 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][i_inner]);
                                __m256 ___x2_4 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][i_inner]);
                                __m256 ___x2_5 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][i_inner]);
                                __m256 ___x2_6 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][i_inner]);
                                __m256 ___x2_7 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][i_inner]);
                                __m256 ___x2_8 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][i_inner]);
                                __m256 ___x2_9 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][i_inner]);
                                __m256 ___x2_10 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][i_inner]);
                                __m256 ___x2_11 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][i_inner]);
                                __m256 ___x2_12 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][i_inner]);
                                __m256 ___x2_13 = _mm256_broadcast_ss(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][i_inner]);
                                ___x0_0 = _mm256_fmadd_ps(___x2_0, ___x1, ___x0_0);
                                ___x0_1 = _mm256_fmadd_ps(___x2_1, ___x1, ___x0_1);
                                ___x0_2 = _mm256_fmadd_ps(___x2_2, ___x1, ___x0_2);
                                ___x0_3 = _mm256_fmadd_ps(___x2_3, ___x1, ___x0_3);
                                ___x0_4 = _mm256_fmadd_ps(___x2_4, ___x1, ___x0_4);
                                ___x0_5 = _mm256_fmadd_ps(___x2_5, ___x1, ___x0_5);
                                ___x0_6 = _mm256_fmadd_ps(___x2_6, ___x1, ___x0_6);
                                ___x0_7 = _mm256_fmadd_ps(___x2_7, ___x1, ___x0_7);
                                ___x0_8 = _mm256_fmadd_ps(___x2_8, ___x1, ___x0_8);
                                ___x0_9 = _mm256_fmadd_ps(___x2_9, ___x1, ___x0_9);
                                ___x0_10 = _mm256_fmadd_ps(___x2_10, ___x1, ___x0_10);
                                ___x0_11 = _mm256_fmadd_ps(___x2_11, ___x1, ___x0_11);
                                ___x0_12 = _mm256_fmadd_ps(___x2_12, ___x1, ___x0_12);
                                ___x0_13 = _mm256_fmadd_ps(___x2_13, ___x1, ___x0_13);
                            }
                        }
                    }
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 0)][0], ___x0_0);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 1)][0], ___x0_1);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 2)][0], ___x0_2);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 3)][0], ___x0_3);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 4)][0], ___x0_4);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 5)][0], ___x0_5);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 6)][0], ___x0_6);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 7)][0], ___x0_7);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 8)][0], ___x0_8);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 9)][0], ___x0_9);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 10)][0], ___x0_10);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 11)][0], ___x0_11);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 12)][0], ___x0_12);
                    _mm256_store_ps(& ensemble2value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][(_neuron_index_3 + 13)][0], ___x0_13);
                }
            }
            for (int _neuron_index_2 = 0; _neuron_index_2 < 56; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 56; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 8; _neuron_index_1_inner += 1) {
                        ensemble3value[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble3inputs[_neuron_index_0][(_neuron_index_1_outer + 1)][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] + ensemble3bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 8; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 56; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 56; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 8; _neuron_index_1_inner += 1) {
                        int _input_offset_1_outer = (_neuron_index_1_outer * 8 + _neuron_index_1_inner - 2 + 8) / 8;
                        int _input_offset_1_inner = (_neuron_index_1_outer * 8 + _neuron_index_1_inner - 2 + 8) % 8;
                        int in_y = _neuron_index_2;
                        int _input_offset_2 = in_y;
                        int in_x = _neuron_index_3;
                        int _input_offset_3 = in_x;
                        long index = ensemble4n[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] / 2;
                        for (int i = 0; i < 5; i += 1) {
                            ensemble4sum_value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble4sum_value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] + ensemble4inputs[_neuron_index_0][((i + _input_offset_1_inner) / 8 + _input_offset_1_outer)][_input_offset_2][_input_offset_3][((i + _input_offset_1_inner) % 8)] * ensemble4inputs[_neuron_index_0][((i + _input_offset_1_inner) / 8 + _input_offset_1_outer)][_input_offset_2][_input_offset_3][((i + _input_offset_1_inner) % 8)];
                        }
                        ensemble4sum_value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble4sum_value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] * (ensemble4alpha[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] / ensemble4n[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner]);
                        ensemble4sum_value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] += ensemble4k[_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        index = ensemble4n[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] / 2;
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble4inputs[_neuron_index_0][((index + _input_offset_1_inner) / 8 + _input_offset_1_outer)][_input_offset_2][_input_offset_3][((index + _input_offset_1_inner) % 8)] / pow(ensemble4sum_value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner], ensemble4beta[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner]);
                    }
                }
            }
        }
    }
    times[0] += omp_get_wtime();
    printf("	times[%d] = %g\n", 0, times[0]);
};
