// <file: forward0.cpp>
#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#if 0
#include "/data/nfs_home/rajbarik/latte/latte/runtime/runtime.h"
#endif
#define SIMDWIDTH 16
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
	__m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
	__m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
	//const int in_width=32, out_width=32;
	if((in_width & 0xf != 0)  || (out_width & 0xf != 0)) {printf("Transpose16x16: Invalid in or out width\n"); return;}

	r0 = _mm512_load_epi32(in + 0*in_width);
	r1 = _mm512_load_epi32(in + 1*in_width);
	r2 = _mm512_load_epi32(in + 2*in_width);
	r3 = _mm512_load_epi32(in + 3*in_width);
	r4 = _mm512_load_epi32(in + 4*in_width);
	r5 = _mm512_load_epi32(in + 5*in_width);
	r6 = _mm512_load_epi32(in + 6*in_width);
	r7 = _mm512_load_epi32(in + 7*in_width);
	r8 = _mm512_load_epi32(in + 8*in_width);
	r9 = _mm512_load_epi32(in + 9*in_width);
	ra = _mm512_load_epi32(in + 10*in_width);
	rb = _mm512_load_epi32(in + 11*in_width);
	rc = _mm512_load_epi32(in + 12*in_width);
	rd = _mm512_load_epi32(in + 13*in_width);
	re = _mm512_load_epi32(in + 14*in_width);
	rf = _mm512_load_epi32(in + 15*in_width);

    t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
    t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...  
    t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
    t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
    ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
    tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
    tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
    td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
    te = _mm512_unpacklo_epi32(re,rf); // 228 ...
    tf = _mm512_unpackhi_epi32(re,rf); // 230 ...

    r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...  
    r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...  
    r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ... 
    rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ... 
    rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

    t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
    t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
    t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
    t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
    t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
    t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
    t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
    t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
    t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
    ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
    tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
    tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
    td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
    te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
    tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
    r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
    r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
    r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
    r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
    r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
    ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
    rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
    rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
    rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
    re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
    rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255

	_mm512_store_epi32(out + 0*out_width, r0);
	_mm512_store_epi32(out + 1*out_width, r1);
	_mm512_store_epi32(out + 2*out_width, r2);
	_mm512_store_epi32(out + 3*out_width, r3);
	_mm512_store_epi32(out + 4*out_width, r4);
	_mm512_store_epi32(out + 5*out_width, r5);
	_mm512_store_epi32(out + 6*out_width, r6);
	_mm512_store_epi32(out + 7*out_width, r7);
	_mm512_store_epi32(out + 8*out_width, r8);
	_mm512_store_epi32(out + 9*out_width, r9);
	_mm512_store_epi32(out + 10*out_width, ra);
	_mm512_store_epi32(out + 11*out_width, rb);
	_mm512_store_epi32(out + 12*out_width, rc);
	_mm512_store_epi32(out + 13*out_width, rd);
	_mm512_store_epi32(out + 14*out_width, re);
	_mm512_store_epi32(out + 15*out_width, rf);
}

extern "C"
void  forward0 (float* _, float* _ensemble2valueensemble2inputs, float* _ensemble2weightsensemble2inputs, float* _ensemble2weights_transposedensemble2inputs, float* _ensemble3biasensemble2inputs, float* _ensemble3inputsensemble2inputs, float* _ensemble3valueensemble2inputs, float* _ensemble4inputsensemble2inputs, float* _ensemble4valueensemble2inputs, float* _ensemble5inputsensemble2inputs, long* _ensemble5mask_jensemble2inputs, long* _ensemble5mask_kensemble2inputs, float* _ensemble5value);
void  forward2 (float* , float* ensemble2weights_transposedensemble4value, float* ensemble3biasensemble4value, float* ensemble5inputsensemble4value, long* ensemble5mask_kensemble4value, float* ensemble4inputsensemble4value, float* ensemble2inputsensemble4value, float* ensemble3inputsensemble4value, long* ensemble5mask_jensemble4value, float* ensemble5valueensemble4value, float* ensemble2valueensemble4value, float* ensemble3value);
void forward0(float* _ensemble2inputs, float* _ensemble2value, float* _ensemble2weights, float* _ensemble2weights_transposed, float* _ensemble3bias, float* _ensemble3inputs, float* _ensemble3value, float* _ensemble4inputs, float* _ensemble4value, float* _ensemble5inputs, long* _ensemble5mask_j, long* _ensemble5mask_k, float* _ensemble5value) {
    float (* ensemble5value)[4][56][56][16] = (float (*)[4][56][56][16]) _ensemble5value;
    __assume_aligned(ensemble5value, 64);
    long (* ensemble5mask_k)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_k;
    __assume_aligned(ensemble5mask_k, 64);
    long (* ensemble5mask_j)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_j;
    __assume_aligned(ensemble5mask_j, 64);
    float (* ensemble5inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble5inputs;
    __assume_aligned(ensemble5inputs, 64);
    float (* ensemble4value)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4value;
    __assume_aligned(ensemble4value, 64);
    float (* ensemble4inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    float (* ensemble3value)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble3value;
    __assume_aligned(ensemble3value, 64);
    float (* ensemble3inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble3inputs;
    __assume_aligned(ensemble3inputs, 64);
    float (* ensemble3bias)[1][16] = (float (*)[1][16]) _ensemble3bias;
    __assume_aligned(ensemble3bias, 64);
    float (* ensemble2weights_transposed)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2weights_transposed;
    __assume_aligned(ensemble2weights_transposed, 64);
    float (* ensemble2weights)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2weights;
    __assume_aligned(ensemble2weights, 64);
    float (* ensemble2value)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble2value;
    __assume_aligned(ensemble2value, 64);
    float (* ensemble2inputs)[1][230][230][16] = (float (*)[1][230][230][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    
    #pragma omp parallel for
    for (int x0 = 0; x0 < 4; x0++) {
      for (int x1 = 0; x1 < 1; x1 ++) {
        for (int x2 = 0; x2 < 7; x2 ++) {
            for (int x3 = 0; x3 < 7; x3 ++) {
                transpose<SIMDWIDTH,SIMDWIDTH>(& ensemble2weights[x0][x1][x2][x3][0][0], & ensemble2weights_transposed[x0][x1][x2][x3][0][0]);
            }
        }
    }
    } 
    forward2(_ensemble4value, _ensemble2weights_transposed, _ensemble3bias, _ensemble5inputs, _ensemble5mask_k, _ensemble4inputs, _ensemble2inputs, _ensemble3inputs, _ensemble5mask_j, _ensemble5value, _ensemble2value, _ensemble3value);
};
void forward2(float* _ensemble4value, float* _ensemble2weights_transposed, float* _ensemble3bias, float* _ensemble5inputs, long* _ensemble5mask_k, float* _ensemble4inputs, float* _ensemble2inputs, float* _ensemble3inputs, long* _ensemble5mask_j, float* _ensemble5value, float* _ensemble2value, float* _ensemble3value) {
    float (* ensemble3value)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble3value;
    __assume_aligned(ensemble3value, 64);
    float (* ensemble2value)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble2value;
    __assume_aligned(ensemble2value, 64);
    float (* ensemble5value)[4][56][56][16] = (float (*)[4][56][56][16]) _ensemble5value;
    __assume_aligned(ensemble5value, 64);
    long (* ensemble5mask_j)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_j;
    __assume_aligned(ensemble5mask_j, 64);
    float (* ensemble3inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble3inputs;
    __assume_aligned(ensemble3inputs, 64);
    float (* ensemble2inputs)[1][230][230][16] = (float (*)[1][230][230][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    float (* ensemble4inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    long (* ensemble5mask_k)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_k;
    __assume_aligned(ensemble5mask_k, 64);
    float (* ensemble5inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble5inputs;
    __assume_aligned(ensemble5inputs, 64);
    float (* ensemble3bias)[1][16] = (float (*)[1][16]) _ensemble3bias;
    __assume_aligned(ensemble3bias, 64);
    float (* ensemble2weights_transposed)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2weights_transposed;
    __assume_aligned(ensemble2weights_transposed, 64);
    float (* ensemble4value)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4value;
    __assume_aligned(ensemble4value, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 4; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 16; _neuron_index_2 += 1) {
                int in_y_0 = (7 * _neuron_index_2 + 0) * 2;
                int in_y_1 = (7 * _neuron_index_2 + 1) * 2;
                int in_y_2 = (7 * _neuron_index_2 + 2) * 2;
                int in_y_3 = (7 * _neuron_index_2 + 3) * 2;
                int in_y_4 = (7 * _neuron_index_2 + 4) * 2;
                int in_y_5 = (7 * _neuron_index_2 + 5) * 2;
                int in_y_6 = (7 * _neuron_index_2 + 6) * 2;
                int _input_offset_2_0 = in_y_0;
                int _input_offset_2_1 = in_y_1;
                int _input_offset_2_2 = in_y_2;
                int _input_offset_2_3 = in_y_3;
                int _input_offset_2_4 = in_y_4;
                int _input_offset_2_5 = in_y_5;
                int _input_offset_2_6 = in_y_6;
                for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 1) {
                    int in_x_0 = (4 * _neuron_index_3 + 0) * 2;
                    int in_x_1 = (4 * _neuron_index_3 + 1) * 2;
                    int in_x_2 = (4 * _neuron_index_3 + 2) * 2;
                    int in_x_3 = (4 * _neuron_index_3 + 3) * 2;
                    int _input_offset_3_0 = in_x_0;
                    int _input_offset_3_1 = in_x_1;
                    int _input_offset_3_2 = in_x_2;
                    int _input_offset_3_3 = in_x_3;
                    __m512 ___x1_0_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_0_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_0_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_0_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x1_1_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_1_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_1_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_1_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x1_2_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_2_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_2_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_2_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x1_3_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_3_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_3_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_3_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x1_4_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_4_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_4_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_4_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x1_5_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_5_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_5_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_5_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x1_6_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x1_6_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x1_6_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x1_6_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 3)][0]);
                    for (int i_outer = 0; i_outer < 1; i_outer += 1) {
                        for (int j = 0; j < 7; j += 1) {
                            for (int k = 0; k < 7; k += 1) {
                                for (int i_inner = 0; i_inner < 16; i_inner += 2) {
                                    __m512 ___x0_0 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                                    __m512 ___x0_1 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                                    __m512 ___x2_0_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_0_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_0_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_0_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_0_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_0_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_0_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_0_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 0) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_1_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_1_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_1_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_1_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_1_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_1_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_1_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_1_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 1) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_2_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_2_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_2_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_2_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_2_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_2_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_2_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_2_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 2) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_3_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_3_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_3_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_3_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_3_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_3_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_3_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_3_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 3) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_4_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_4_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_4_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_4_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_4_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_4_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_4_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_4_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 4) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_5_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_5_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_5_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_5_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_5_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_5_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_5_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_5_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 5) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_6_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_6_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 0) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_6_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_6_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 1) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_6_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_6_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 2) * 2)][(i_inner + 1)]);
                                    __m512 ___x2_6_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 0)]);
                                    __m512 ___x2_6_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + (7 * _neuron_index_2 + 6) * 2)][(k * 1 + (4 * _neuron_index_3 + 3) * 2)][(i_inner + 1)]);
                                    ___x1_0_0 = _mm512_fmadd_ps(___x2_0_0_0, ___x0_0, ___x1_0_0);
                                    ___x1_0_0 = _mm512_fmadd_ps(___x2_0_0_1, ___x0_1, ___x1_0_0);
                                    ___x1_0_1 = _mm512_fmadd_ps(___x2_0_1_0, ___x0_0, ___x1_0_1);
                                    ___x1_0_1 = _mm512_fmadd_ps(___x2_0_1_1, ___x0_1, ___x1_0_1);
                                    ___x1_0_2 = _mm512_fmadd_ps(___x2_0_2_0, ___x0_0, ___x1_0_2);
                                    ___x1_0_2 = _mm512_fmadd_ps(___x2_0_2_1, ___x0_1, ___x1_0_2);
                                    ___x1_0_3 = _mm512_fmadd_ps(___x2_0_3_0, ___x0_0, ___x1_0_3);
                                    ___x1_0_3 = _mm512_fmadd_ps(___x2_0_3_1, ___x0_1, ___x1_0_3);
                                    ___x1_1_0 = _mm512_fmadd_ps(___x2_1_0_0, ___x0_0, ___x1_1_0);
                                    ___x1_1_0 = _mm512_fmadd_ps(___x2_1_0_1, ___x0_1, ___x1_1_0);
                                    ___x1_1_1 = _mm512_fmadd_ps(___x2_1_1_0, ___x0_0, ___x1_1_1);
                                    ___x1_1_1 = _mm512_fmadd_ps(___x2_1_1_1, ___x0_1, ___x1_1_1);
                                    ___x1_1_2 = _mm512_fmadd_ps(___x2_1_2_0, ___x0_0, ___x1_1_2);
                                    ___x1_1_2 = _mm512_fmadd_ps(___x2_1_2_1, ___x0_1, ___x1_1_2);
                                    ___x1_1_3 = _mm512_fmadd_ps(___x2_1_3_0, ___x0_0, ___x1_1_3);
                                    ___x1_1_3 = _mm512_fmadd_ps(___x2_1_3_1, ___x0_1, ___x1_1_3);
                                    ___x1_2_0 = _mm512_fmadd_ps(___x2_2_0_0, ___x0_0, ___x1_2_0);
                                    ___x1_2_0 = _mm512_fmadd_ps(___x2_2_0_1, ___x0_1, ___x1_2_0);
                                    ___x1_2_1 = _mm512_fmadd_ps(___x2_2_1_0, ___x0_0, ___x1_2_1);
                                    ___x1_2_1 = _mm512_fmadd_ps(___x2_2_1_1, ___x0_1, ___x1_2_1);
                                    ___x1_2_2 = _mm512_fmadd_ps(___x2_2_2_0, ___x0_0, ___x1_2_2);
                                    ___x1_2_2 = _mm512_fmadd_ps(___x2_2_2_1, ___x0_1, ___x1_2_2);
                                    ___x1_2_3 = _mm512_fmadd_ps(___x2_2_3_0, ___x0_0, ___x1_2_3);
                                    ___x1_2_3 = _mm512_fmadd_ps(___x2_2_3_1, ___x0_1, ___x1_2_3);
                                    ___x1_3_0 = _mm512_fmadd_ps(___x2_3_0_0, ___x0_0, ___x1_3_0);
                                    ___x1_3_0 = _mm512_fmadd_ps(___x2_3_0_1, ___x0_1, ___x1_3_0);
                                    ___x1_3_1 = _mm512_fmadd_ps(___x2_3_1_0, ___x0_0, ___x1_3_1);
                                    ___x1_3_1 = _mm512_fmadd_ps(___x2_3_1_1, ___x0_1, ___x1_3_1);
                                    ___x1_3_2 = _mm512_fmadd_ps(___x2_3_2_0, ___x0_0, ___x1_3_2);
                                    ___x1_3_2 = _mm512_fmadd_ps(___x2_3_2_1, ___x0_1, ___x1_3_2);
                                    ___x1_3_3 = _mm512_fmadd_ps(___x2_3_3_0, ___x0_0, ___x1_3_3);
                                    ___x1_3_3 = _mm512_fmadd_ps(___x2_3_3_1, ___x0_1, ___x1_3_3);
                                    ___x1_4_0 = _mm512_fmadd_ps(___x2_4_0_0, ___x0_0, ___x1_4_0);
                                    ___x1_4_0 = _mm512_fmadd_ps(___x2_4_0_1, ___x0_1, ___x1_4_0);
                                    ___x1_4_1 = _mm512_fmadd_ps(___x2_4_1_0, ___x0_0, ___x1_4_1);
                                    ___x1_4_1 = _mm512_fmadd_ps(___x2_4_1_1, ___x0_1, ___x1_4_1);
                                    ___x1_4_2 = _mm512_fmadd_ps(___x2_4_2_0, ___x0_0, ___x1_4_2);
                                    ___x1_4_2 = _mm512_fmadd_ps(___x2_4_2_1, ___x0_1, ___x1_4_2);
                                    ___x1_4_3 = _mm512_fmadd_ps(___x2_4_3_0, ___x0_0, ___x1_4_3);
                                    ___x1_4_3 = _mm512_fmadd_ps(___x2_4_3_1, ___x0_1, ___x1_4_3);
                                    ___x1_5_0 = _mm512_fmadd_ps(___x2_5_0_0, ___x0_0, ___x1_5_0);
                                    ___x1_5_0 = _mm512_fmadd_ps(___x2_5_0_1, ___x0_1, ___x1_5_0);
                                    ___x1_5_1 = _mm512_fmadd_ps(___x2_5_1_0, ___x0_0, ___x1_5_1);
                                    ___x1_5_1 = _mm512_fmadd_ps(___x2_5_1_1, ___x0_1, ___x1_5_1);
                                    ___x1_5_2 = _mm512_fmadd_ps(___x2_5_2_0, ___x0_0, ___x1_5_2);
                                    ___x1_5_2 = _mm512_fmadd_ps(___x2_5_2_1, ___x0_1, ___x1_5_2);
                                    ___x1_5_3 = _mm512_fmadd_ps(___x2_5_3_0, ___x0_0, ___x1_5_3);
                                    ___x1_5_3 = _mm512_fmadd_ps(___x2_5_3_1, ___x0_1, ___x1_5_3);
                                    ___x1_6_0 = _mm512_fmadd_ps(___x2_6_0_0, ___x0_0, ___x1_6_0);
                                    ___x1_6_0 = _mm512_fmadd_ps(___x2_6_0_1, ___x0_1, ___x1_6_0);
                                    ___x1_6_1 = _mm512_fmadd_ps(___x2_6_1_0, ___x0_0, ___x1_6_1);
                                    ___x1_6_1 = _mm512_fmadd_ps(___x2_6_1_1, ___x0_1, ___x1_6_1);
                                    ___x1_6_2 = _mm512_fmadd_ps(___x2_6_2_0, ___x0_0, ___x1_6_2);
                                    ___x1_6_2 = _mm512_fmadd_ps(___x2_6_2_1, ___x0_1, ___x1_6_2);
                                    ___x1_6_3 = _mm512_fmadd_ps(___x2_6_3_0, ___x0_0, ___x1_6_3);
                                    ___x1_6_3 = _mm512_fmadd_ps(___x2_6_3_1, ___x0_1, ___x1_6_3);
                                }
                            }
                        }
                    }
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 0)][0], ___x1_0_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 1)][0], ___x1_0_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 2)][0], ___x1_0_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 3)][0], ___x1_0_3);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 0)][0], ___x1_1_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 1)][0], ___x1_1_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 2)][0], ___x1_1_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 3)][0], ___x1_1_3);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 0)][0], ___x1_2_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 1)][0], ___x1_2_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 2)][0], ___x1_2_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 3)][0], ___x1_2_3);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 0)][0], ___x1_3_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 1)][0], ___x1_3_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 2)][0], ___x1_3_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 3)][0], ___x1_3_3);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 0)][0], ___x1_4_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 1)][0], ___x1_4_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 2)][0], ___x1_4_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 3)][0], ___x1_4_3);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 0)][0], ___x1_5_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 1)][0], ___x1_5_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 2)][0], ___x1_5_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 3)][0], ___x1_5_3);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 0)][0], ___x1_6_0);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 1)][0], ___x1_6_1);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 2)][0], ___x1_6_2);
                    _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 3)][0], ___x1_6_3);
            __m512 ___x10 = _mm512_load_ps(& ensemble3bias[_neuron_index_1_outer][0][0]);
                    __m512 ___x9_0_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_0_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_0_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_0_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x9_1_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_1_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_1_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_1_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x9_2_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_2_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_2_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_2_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x9_3_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_3_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_3_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_3_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x9_4_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_4_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_4_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_4_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x9_5_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_5_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_5_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_5_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 3)][0]);
                    __m512 ___x9_6_0 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 0)][0]);
                    __m512 ___x9_6_1 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 1)][0]);
                    __m512 ___x9_6_2 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 2)][0]);
                    __m512 ___x9_6_3 = _mm512_load_ps(& ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 3)][0]);
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_0_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_0_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_0_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_0_3, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_1_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_1_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_1_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_1_3, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_2_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_2_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_2_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_2_3, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_3_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_3_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_3_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_3_3, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_4_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_4_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_4_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_4_3, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_5_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_5_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_5_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_5_3, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 0)][0], _mm512_add_ps(___x9_6_0, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 1)][0], _mm512_add_ps(___x9_6_1, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 2)][0], _mm512_add_ps(___x9_6_2, ___x10));
                    _mm512_store_ps(& ensemble3value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 3)][0], _mm512_add_ps(___x9_6_3, ___x10));
                    #pragma simd
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 0)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 1)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 2)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 3)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 4)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 5)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 0)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 2)][_neuron_index_1_inner], (float) 0.0);
                        ensemble4value[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(7 * _neuron_index_2 + 6)][(4 * _neuron_index_3 + 3)][_neuron_index_1_inner], (float) 0.0);
                    }
                }
            }
            for (int _neuron_index_2 = 0; _neuron_index_2 < 56; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 56; _neuron_index_3 += 1) {
                    #pragma simd
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        int _input_offset_1_outer = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) / 16;
                        int _input_offset_1_inner = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) % 16;
                        int in_y = _neuron_index_2 * 2 - 0;
                        int _input_offset_2 = in_y;
                        int in_x = _neuron_index_3 * 2 - 0;
                        int _input_offset_3 = in_x;
                        float max_value = - INFINITY;
                        for (int j = 0; j < 3; j += 1) {
                            for (int k = 0; k < 3; k += 1) {
                                if (ensemble5inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j * 1 + _input_offset_2, 0), 111)][MIN(MAX(k * 1 + _input_offset_3, 0), 111)][_input_offset_1_inner] > max_value) {
                                    max_value = ensemble5inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j * 1 + _input_offset_2, 0), 111)][MIN(MAX(k * 1 + _input_offset_3, 0), 111)][_input_offset_1_inner];
                                    ensemble5mask_j[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = j;
                                    ensemble5mask_k[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = k;
                                };
                            }
                        }
                        ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = max_value;
                    }
                }
            }
        }
    }
};
