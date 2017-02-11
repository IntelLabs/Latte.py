// <file: forward0.cpp>
#include <immintrin.h>
#include <mkl.h>
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#if 1
#include "/data/nfs_home/avenkat/latte/latte/runtime/runtime.h"
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
void forward0(float* _ensemble2inputs, float* _ensemble2value, float* _ensemble2weights, float* _ensemble2weights_transposed, float* _ensemble3bias, float* _ensemble3inputs, float* _ensemble3value, float* _ensemble4inputs, float* _ensemble4value, float* _ensemble5inputs, float* _ensemble5value, float* _ensemble5weights, float* _ensemble5weights_transposed, float* _ensemble6bias, float* _ensemble6inputs, float* _ensemble6value, float* _ensemble7inputs, float* _ensemble7value, float* _ensemble8inputs, long* _ensemble8mask_j, long* _ensemble8mask_k, float* _ensemble8value) {
    float (* ensemble8value)[8][14][14][16] = (float (*)[8][14][14][16]) _ensemble8value;
    __assume_aligned(ensemble8value, 64);
    long (* ensemble8mask_k)[8][14][14][16] = (long (*)[8][14][14][16]) _ensemble8mask_k;
    __assume_aligned(ensemble8mask_k, 64);
    long (* ensemble8mask_j)[8][14][14][16] = (long (*)[8][14][14][16]) _ensemble8mask_j;
    __assume_aligned(ensemble8mask_j, 64);
    float (* ensemble8inputs)[8][28][28][16] = (float (*)[8][28][28][16]) _ensemble8inputs;
    __assume_aligned(ensemble8inputs, 64);
    float (* ensemble7value)[8][28][28][16] = (float (*)[8][28][28][16]) _ensemble7value;
    __assume_aligned(ensemble7value, 64);
    float (* ensemble7inputs)[8][28][28][16] = (float (*)[8][28][28][16]) _ensemble7inputs;
    __assume_aligned(ensemble7inputs, 64);
    float (* ensemble6value)[8][28][28][16] = (float (*)[8][28][28][16]) _ensemble6value;
    __assume_aligned(ensemble6value, 64);
    float (* ensemble6inputs)[8][28][28][16] = (float (*)[8][28][28][16]) _ensemble6inputs;
    __assume_aligned(ensemble6inputs, 64);
    float (* ensemble6bias)[1][16] = (float (*)[1][16]) _ensemble6bias;
    __assume_aligned(ensemble6bias, 64);
    float (* ensemble5weights_transposed)[6][3][3][16][16] = (float (*)[6][3][3][16][16]) _ensemble5weights_transposed;
    __assume_aligned(ensemble5weights_transposed, 64);
    float (* ensemble5weights)[6][3][3][16][16] = (float (*)[6][3][3][16][16]) _ensemble5weights;
    __assume_aligned(ensemble5weights, 64);
    float (* ensemble5value)[8][28][28][16] = (float (*)[8][28][28][16]) _ensemble5value;
    __assume_aligned(ensemble5value, 64);
    float (* ensemble5inputs)[6][30][30][16] = (float (*)[6][30][30][16]) _ensemble5inputs;
    __assume_aligned(ensemble5inputs, 64);
    float (* ensemble4value)[6][30][30][16] = (float (*)[6][30][30][16]) _ensemble4value;
    __assume_aligned(ensemble4value, 64);
    float (* ensemble4inputs)[6][30][30][16] = (float (*)[6][30][30][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    float (* ensemble3value)[6][30][30][16] = (float (*)[6][30][30][16]) _ensemble3value;
    __assume_aligned(ensemble3value, 64);
    float (* ensemble3inputs)[6][30][30][16] = (float (*)[6][30][30][16]) _ensemble3inputs;
    __assume_aligned(ensemble3inputs, 64);
    float (* ensemble3bias)[1][16] = (float (*)[1][16]) _ensemble3bias;
    __assume_aligned(ensemble3bias, 64);
    float (* ensemble2weights_transposed)[1][1][1][16][16] = (float (*)[1][1][1][16][16]) _ensemble2weights_transposed;
    __assume_aligned(ensemble2weights_transposed, 64);
    float (* ensemble2weights)[1][1][1][16][16] = (float (*)[1][1][1][16][16]) _ensemble2weights;
    __assume_aligned(ensemble2weights, 64);
    float (* ensemble2value)[6][30][30][16] = (float (*)[6][30][30][16]) _ensemble2value;
    __assume_aligned(ensemble2value, 64);
    float (* ensemble2inputs)[1][28][28][16] = (float (*)[1][28][28][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    
    parallel_for(0, 6,
      [=](int low, int high) {
        for (int x0 = low; x0 < high; x0++) {
          for (int x1 = 0; x1 < 1; x1 ++) {
        for (int x2 = 0; x2 < 1; x2 ++) {
            for (int x3 = 0; x3 < 1; x3 ++) {
                transpose<SIMDWIDTH,SIMDWIDTH>(& ensemble2weights[x0][x1][x2][x3][0][0], & ensemble2weights_transposed[x0][x1][x2][x3][0][0]);
            }
        }
    }
        } 
      }
    );
    
    parallel_for(0,128 / 1,
      [=](int low, int high) {
        for (int tmp__neuron_index_0 = low; tmp__neuron_index_0 < high; tmp__neuron_index_0++) {
          int _neuron_index_0 = tmp__neuron_index_0 * 1;
          
    parallel_for(0,6 / 1,
      [=](int low, int high) {
        for (int tmp__neuron_index_1_outer = low; tmp__neuron_index_1_outer < high; tmp__neuron_index_1_outer++) {
          int _neuron_index_1_outer = tmp__neuron_index_1_outer * 1;
          for (int i_outer = 0; i_outer < 1; i_outer += 1) {
        for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
            int in_y = _neuron_index_2 * 1;
            int _input_offset_2 = in_y;
            for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 28) {
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
                int in_x_14 = (_neuron_index_3 + 14) * 1;
                int in_x_15 = (_neuron_index_3 + 15) * 1;
                int in_x_16 = (_neuron_index_3 + 16) * 1;
                int in_x_17 = (_neuron_index_3 + 17) * 1;
                int in_x_18 = (_neuron_index_3 + 18) * 1;
                int in_x_19 = (_neuron_index_3 + 19) * 1;
                int in_x_20 = (_neuron_index_3 + 20) * 1;
                int in_x_21 = (_neuron_index_3 + 21) * 1;
                int in_x_22 = (_neuron_index_3 + 22) * 1;
                int in_x_23 = (_neuron_index_3 + 23) * 1;
                int in_x_24 = (_neuron_index_3 + 24) * 1;
                int in_x_25 = (_neuron_index_3 + 25) * 1;
                int in_x_26 = (_neuron_index_3 + 26) * 1;
                int in_x_27 = (_neuron_index_3 + 27) * 1;
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
                int _input_offset_3_14 = in_x_14;
                int _input_offset_3_15 = in_x_15;
                int _input_offset_3_16 = in_x_16;
                int _input_offset_3_17 = in_x_17;
                int _input_offset_3_18 = in_x_18;
                int _input_offset_3_19 = in_x_19;
                int _input_offset_3_20 = in_x_20;
                int _input_offset_3_21 = in_x_21;
                int _input_offset_3_22 = in_x_22;
                int _input_offset_3_23 = in_x_23;
                int _input_offset_3_24 = in_x_24;
                int _input_offset_3_25 = in_x_25;
                int _input_offset_3_26 = in_x_26;
                int _input_offset_3_27 = in_x_27;
                __m512 ___x1_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 0 + 1)][0]);
                __m512 ___x1_1 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1 + 1)][0]);
                __m512 ___x1_2 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 2 + 1)][0]);
                __m512 ___x1_3 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 3 + 1)][0]);
                __m512 ___x1_4 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 4 + 1)][0]);
                __m512 ___x1_5 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 5 + 1)][0]);
                __m512 ___x1_6 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 6 + 1)][0]);
                __m512 ___x1_7 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 7 + 1)][0]);
                __m512 ___x1_8 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 8 + 1)][0]);
                __m512 ___x1_9 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 9 + 1)][0]);
                __m512 ___x1_10 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 10 + 1)][0]);
                __m512 ___x1_11 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 11 + 1)][0]);
                __m512 ___x1_12 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 12 + 1)][0]);
                __m512 ___x1_13 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 13 + 1)][0]);
                __m512 ___x1_14 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 14 + 1)][0]);
                __m512 ___x1_15 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 15 + 1)][0]);
                __m512 ___x1_16 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 16 + 1)][0]);
                __m512 ___x1_17 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 17 + 1)][0]);
                __m512 ___x1_18 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 18 + 1)][0]);
                __m512 ___x1_19 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 19 + 1)][0]);
                __m512 ___x1_20 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 20 + 1)][0]);
                __m512 ___x1_21 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 21 + 1)][0]);
                __m512 ___x1_22 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 22 + 1)][0]);
                __m512 ___x1_23 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 23 + 1)][0]);
                __m512 ___x1_24 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 24 + 1)][0]);
                __m512 ___x1_25 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 25 + 1)][0]);
                __m512 ___x1_26 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 26 + 1)][0]);
                __m512 ___x1_27 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 27 + 1)][0]);
                for (int j = 0; j < 1; j += 1) {
                    for (int k = 0; k < 1; k += 1) {
                        for (int i_inner = 0; i_inner < 16; i_inner += 4) {
                            __m512 ___x0_0 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                            __m512 ___x0_1 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                            __m512 ___x0_2 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 2)][0]);
                            __m512 ___x0_3 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 3)][0]);
                            __m512 ___x2_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 0)]);
                            __m512 ___x2_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 1)]);
                            __m512 ___x2_0_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 2)]);
                            __m512 ___x2_0_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 3)]);
                            __m512 ___x2_1_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 0)]);
                            __m512 ___x2_1_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 1)]);
                            __m512 ___x2_1_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 2)]);
                            __m512 ___x2_1_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 3)]);
                            __m512 ___x2_2_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 0)]);
                            __m512 ___x2_2_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 1)]);
                            __m512 ___x2_2_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 2)]);
                            __m512 ___x2_2_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 3)]);
                            __m512 ___x2_3_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 0)]);
                            __m512 ___x2_3_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 1)]);
                            __m512 ___x2_3_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 2)]);
                            __m512 ___x2_3_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 3)]);
                            __m512 ___x2_4_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 0)]);
                            __m512 ___x2_4_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 1)]);
                            __m512 ___x2_4_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 2)]);
                            __m512 ___x2_4_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 3)]);
                            __m512 ___x2_5_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 0)]);
                            __m512 ___x2_5_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 1)]);
                            __m512 ___x2_5_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 2)]);
                            __m512 ___x2_5_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 3)]);
                            __m512 ___x2_6_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 0)]);
                            __m512 ___x2_6_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 1)]);
                            __m512 ___x2_6_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 2)]);
                            __m512 ___x2_6_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 3)]);
                            __m512 ___x2_7_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 0)]);
                            __m512 ___x2_7_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 1)]);
                            __m512 ___x2_7_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 2)]);
                            __m512 ___x2_7_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 3)]);
                            __m512 ___x2_8_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 0)]);
                            __m512 ___x2_8_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 1)]);
                            __m512 ___x2_8_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 2)]);
                            __m512 ___x2_8_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 3)]);
                            __m512 ___x2_9_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 0)]);
                            __m512 ___x2_9_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 1)]);
                            __m512 ___x2_9_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 2)]);
                            __m512 ___x2_9_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 3)]);
                            __m512 ___x2_10_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 0)]);
                            __m512 ___x2_10_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 1)]);
                            __m512 ___x2_10_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 2)]);
                            __m512 ___x2_10_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 3)]);
                            __m512 ___x2_11_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 0)]);
                            __m512 ___x2_11_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 1)]);
                            __m512 ___x2_11_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 2)]);
                            __m512 ___x2_11_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 3)]);
                            __m512 ___x2_12_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 0)]);
                            __m512 ___x2_12_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 1)]);
                            __m512 ___x2_12_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 2)]);
                            __m512 ___x2_12_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 3)]);
                            __m512 ___x2_13_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 0)]);
                            __m512 ___x2_13_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 1)]);
                            __m512 ___x2_13_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 2)]);
                            __m512 ___x2_13_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 3)]);
                            __m512 ___x2_14_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 0)]);
                            __m512 ___x2_14_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 1)]);
                            __m512 ___x2_14_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 2)]);
                            __m512 ___x2_14_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 3)]);
                            __m512 ___x2_15_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 0)]);
                            __m512 ___x2_15_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 1)]);
                            __m512 ___x2_15_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 2)]);
                            __m512 ___x2_15_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 3)]);
                            __m512 ___x2_16_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 0)]);
                            __m512 ___x2_16_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 1)]);
                            __m512 ___x2_16_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 2)]);
                            __m512 ___x2_16_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 3)]);
                            __m512 ___x2_17_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 0)]);
                            __m512 ___x2_17_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 1)]);
                            __m512 ___x2_17_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 2)]);
                            __m512 ___x2_17_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 3)]);
                            __m512 ___x2_18_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 0)]);
                            __m512 ___x2_18_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 1)]);
                            __m512 ___x2_18_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 2)]);
                            __m512 ___x2_18_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 3)]);
                            __m512 ___x2_19_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 0)]);
                            __m512 ___x2_19_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 1)]);
                            __m512 ___x2_19_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 2)]);
                            __m512 ___x2_19_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 3)]);
                            __m512 ___x2_20_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 0)]);
                            __m512 ___x2_20_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 1)]);
                            __m512 ___x2_20_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 2)]);
                            __m512 ___x2_20_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 3)]);
                            __m512 ___x2_21_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 0)]);
                            __m512 ___x2_21_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 1)]);
                            __m512 ___x2_21_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 2)]);
                            __m512 ___x2_21_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 3)]);
                            __m512 ___x2_22_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 0)]);
                            __m512 ___x2_22_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 1)]);
                            __m512 ___x2_22_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 2)]);
                            __m512 ___x2_22_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 3)]);
                            __m512 ___x2_23_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 0)]);
                            __m512 ___x2_23_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 1)]);
                            __m512 ___x2_23_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 2)]);
                            __m512 ___x2_23_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 3)]);
                            __m512 ___x2_24_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 0)]);
                            __m512 ___x2_24_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 1)]);
                            __m512 ___x2_24_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 2)]);
                            __m512 ___x2_24_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 3)]);
                            __m512 ___x2_25_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 0)]);
                            __m512 ___x2_25_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 1)]);
                            __m512 ___x2_25_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 2)]);
                            __m512 ___x2_25_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 3)]);
                            __m512 ___x2_26_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 0)]);
                            __m512 ___x2_26_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 1)]);
                            __m512 ___x2_26_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 2)]);
                            __m512 ___x2_26_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 3)]);
                            __m512 ___x2_27_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 0)]);
                            __m512 ___x2_27_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 1)]);
                            __m512 ___x2_27_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 2)]);
                            __m512 ___x2_27_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 3)]);
                            ___x1_0 = _mm512_fmadd_ps(___x2_0_0, ___x0_0, ___x1_0);
                            ___x1_0 = _mm512_fmadd_ps(___x2_0_1, ___x0_1, ___x1_0);
                            ___x1_0 = _mm512_fmadd_ps(___x2_0_2, ___x0_2, ___x1_0);
                            ___x1_0 = _mm512_fmadd_ps(___x2_0_3, ___x0_3, ___x1_0);
                            ___x1_1 = _mm512_fmadd_ps(___x2_1_0, ___x0_0, ___x1_1);
                            ___x1_1 = _mm512_fmadd_ps(___x2_1_1, ___x0_1, ___x1_1);
                            ___x1_1 = _mm512_fmadd_ps(___x2_1_2, ___x0_2, ___x1_1);
                            ___x1_1 = _mm512_fmadd_ps(___x2_1_3, ___x0_3, ___x1_1);
                            ___x1_2 = _mm512_fmadd_ps(___x2_2_0, ___x0_0, ___x1_2);
                            ___x1_2 = _mm512_fmadd_ps(___x2_2_1, ___x0_1, ___x1_2);
                            ___x1_2 = _mm512_fmadd_ps(___x2_2_2, ___x0_2, ___x1_2);
                            ___x1_2 = _mm512_fmadd_ps(___x2_2_3, ___x0_3, ___x1_2);
                            ___x1_3 = _mm512_fmadd_ps(___x2_3_0, ___x0_0, ___x1_3);
                            ___x1_3 = _mm512_fmadd_ps(___x2_3_1, ___x0_1, ___x1_3);
                            ___x1_3 = _mm512_fmadd_ps(___x2_3_2, ___x0_2, ___x1_3);
                            ___x1_3 = _mm512_fmadd_ps(___x2_3_3, ___x0_3, ___x1_3);
                            ___x1_4 = _mm512_fmadd_ps(___x2_4_0, ___x0_0, ___x1_4);
                            ___x1_4 = _mm512_fmadd_ps(___x2_4_1, ___x0_1, ___x1_4);
                            ___x1_4 = _mm512_fmadd_ps(___x2_4_2, ___x0_2, ___x1_4);
                            ___x1_4 = _mm512_fmadd_ps(___x2_4_3, ___x0_3, ___x1_4);
                            ___x1_5 = _mm512_fmadd_ps(___x2_5_0, ___x0_0, ___x1_5);
                            ___x1_5 = _mm512_fmadd_ps(___x2_5_1, ___x0_1, ___x1_5);
                            ___x1_5 = _mm512_fmadd_ps(___x2_5_2, ___x0_2, ___x1_5);
                            ___x1_5 = _mm512_fmadd_ps(___x2_5_3, ___x0_3, ___x1_5);
                            ___x1_6 = _mm512_fmadd_ps(___x2_6_0, ___x0_0, ___x1_6);
                            ___x1_6 = _mm512_fmadd_ps(___x2_6_1, ___x0_1, ___x1_6);
                            ___x1_6 = _mm512_fmadd_ps(___x2_6_2, ___x0_2, ___x1_6);
                            ___x1_6 = _mm512_fmadd_ps(___x2_6_3, ___x0_3, ___x1_6);
                            ___x1_7 = _mm512_fmadd_ps(___x2_7_0, ___x0_0, ___x1_7);
                            ___x1_7 = _mm512_fmadd_ps(___x2_7_1, ___x0_1, ___x1_7);
                            ___x1_7 = _mm512_fmadd_ps(___x2_7_2, ___x0_2, ___x1_7);
                            ___x1_7 = _mm512_fmadd_ps(___x2_7_3, ___x0_3, ___x1_7);
                            ___x1_8 = _mm512_fmadd_ps(___x2_8_0, ___x0_0, ___x1_8);
                            ___x1_8 = _mm512_fmadd_ps(___x2_8_1, ___x0_1, ___x1_8);
                            ___x1_8 = _mm512_fmadd_ps(___x2_8_2, ___x0_2, ___x1_8);
                            ___x1_8 = _mm512_fmadd_ps(___x2_8_3, ___x0_3, ___x1_8);
                            ___x1_9 = _mm512_fmadd_ps(___x2_9_0, ___x0_0, ___x1_9);
                            ___x1_9 = _mm512_fmadd_ps(___x2_9_1, ___x0_1, ___x1_9);
                            ___x1_9 = _mm512_fmadd_ps(___x2_9_2, ___x0_2, ___x1_9);
                            ___x1_9 = _mm512_fmadd_ps(___x2_9_3, ___x0_3, ___x1_9);
                            ___x1_10 = _mm512_fmadd_ps(___x2_10_0, ___x0_0, ___x1_10);
                            ___x1_10 = _mm512_fmadd_ps(___x2_10_1, ___x0_1, ___x1_10);
                            ___x1_10 = _mm512_fmadd_ps(___x2_10_2, ___x0_2, ___x1_10);
                            ___x1_10 = _mm512_fmadd_ps(___x2_10_3, ___x0_3, ___x1_10);
                            ___x1_11 = _mm512_fmadd_ps(___x2_11_0, ___x0_0, ___x1_11);
                            ___x1_11 = _mm512_fmadd_ps(___x2_11_1, ___x0_1, ___x1_11);
                            ___x1_11 = _mm512_fmadd_ps(___x2_11_2, ___x0_2, ___x1_11);
                            ___x1_11 = _mm512_fmadd_ps(___x2_11_3, ___x0_3, ___x1_11);
                            ___x1_12 = _mm512_fmadd_ps(___x2_12_0, ___x0_0, ___x1_12);
                            ___x1_12 = _mm512_fmadd_ps(___x2_12_1, ___x0_1, ___x1_12);
                            ___x1_12 = _mm512_fmadd_ps(___x2_12_2, ___x0_2, ___x1_12);
                            ___x1_12 = _mm512_fmadd_ps(___x2_12_3, ___x0_3, ___x1_12);
                            ___x1_13 = _mm512_fmadd_ps(___x2_13_0, ___x0_0, ___x1_13);
                            ___x1_13 = _mm512_fmadd_ps(___x2_13_1, ___x0_1, ___x1_13);
                            ___x1_13 = _mm512_fmadd_ps(___x2_13_2, ___x0_2, ___x1_13);
                            ___x1_13 = _mm512_fmadd_ps(___x2_13_3, ___x0_3, ___x1_13);
                            ___x1_14 = _mm512_fmadd_ps(___x2_14_0, ___x0_0, ___x1_14);
                            ___x1_14 = _mm512_fmadd_ps(___x2_14_1, ___x0_1, ___x1_14);
                            ___x1_14 = _mm512_fmadd_ps(___x2_14_2, ___x0_2, ___x1_14);
                            ___x1_14 = _mm512_fmadd_ps(___x2_14_3, ___x0_3, ___x1_14);
                            ___x1_15 = _mm512_fmadd_ps(___x2_15_0, ___x0_0, ___x1_15);
                            ___x1_15 = _mm512_fmadd_ps(___x2_15_1, ___x0_1, ___x1_15);
                            ___x1_15 = _mm512_fmadd_ps(___x2_15_2, ___x0_2, ___x1_15);
                            ___x1_15 = _mm512_fmadd_ps(___x2_15_3, ___x0_3, ___x1_15);
                            ___x1_16 = _mm512_fmadd_ps(___x2_16_0, ___x0_0, ___x1_16);
                            ___x1_16 = _mm512_fmadd_ps(___x2_16_1, ___x0_1, ___x1_16);
                            ___x1_16 = _mm512_fmadd_ps(___x2_16_2, ___x0_2, ___x1_16);
                            ___x1_16 = _mm512_fmadd_ps(___x2_16_3, ___x0_3, ___x1_16);
                            ___x1_17 = _mm512_fmadd_ps(___x2_17_0, ___x0_0, ___x1_17);
                            ___x1_17 = _mm512_fmadd_ps(___x2_17_1, ___x0_1, ___x1_17);
                            ___x1_17 = _mm512_fmadd_ps(___x2_17_2, ___x0_2, ___x1_17);
                            ___x1_17 = _mm512_fmadd_ps(___x2_17_3, ___x0_3, ___x1_17);
                            ___x1_18 = _mm512_fmadd_ps(___x2_18_0, ___x0_0, ___x1_18);
                            ___x1_18 = _mm512_fmadd_ps(___x2_18_1, ___x0_1, ___x1_18);
                            ___x1_18 = _mm512_fmadd_ps(___x2_18_2, ___x0_2, ___x1_18);
                            ___x1_18 = _mm512_fmadd_ps(___x2_18_3, ___x0_3, ___x1_18);
                            ___x1_19 = _mm512_fmadd_ps(___x2_19_0, ___x0_0, ___x1_19);
                            ___x1_19 = _mm512_fmadd_ps(___x2_19_1, ___x0_1, ___x1_19);
                            ___x1_19 = _mm512_fmadd_ps(___x2_19_2, ___x0_2, ___x1_19);
                            ___x1_19 = _mm512_fmadd_ps(___x2_19_3, ___x0_3, ___x1_19);
                            ___x1_20 = _mm512_fmadd_ps(___x2_20_0, ___x0_0, ___x1_20);
                            ___x1_20 = _mm512_fmadd_ps(___x2_20_1, ___x0_1, ___x1_20);
                            ___x1_20 = _mm512_fmadd_ps(___x2_20_2, ___x0_2, ___x1_20);
                            ___x1_20 = _mm512_fmadd_ps(___x2_20_3, ___x0_3, ___x1_20);
                            ___x1_21 = _mm512_fmadd_ps(___x2_21_0, ___x0_0, ___x1_21);
                            ___x1_21 = _mm512_fmadd_ps(___x2_21_1, ___x0_1, ___x1_21);
                            ___x1_21 = _mm512_fmadd_ps(___x2_21_2, ___x0_2, ___x1_21);
                            ___x1_21 = _mm512_fmadd_ps(___x2_21_3, ___x0_3, ___x1_21);
                            ___x1_22 = _mm512_fmadd_ps(___x2_22_0, ___x0_0, ___x1_22);
                            ___x1_22 = _mm512_fmadd_ps(___x2_22_1, ___x0_1, ___x1_22);
                            ___x1_22 = _mm512_fmadd_ps(___x2_22_2, ___x0_2, ___x1_22);
                            ___x1_22 = _mm512_fmadd_ps(___x2_22_3, ___x0_3, ___x1_22);
                            ___x1_23 = _mm512_fmadd_ps(___x2_23_0, ___x0_0, ___x1_23);
                            ___x1_23 = _mm512_fmadd_ps(___x2_23_1, ___x0_1, ___x1_23);
                            ___x1_23 = _mm512_fmadd_ps(___x2_23_2, ___x0_2, ___x1_23);
                            ___x1_23 = _mm512_fmadd_ps(___x2_23_3, ___x0_3, ___x1_23);
                            ___x1_24 = _mm512_fmadd_ps(___x2_24_0, ___x0_0, ___x1_24);
                            ___x1_24 = _mm512_fmadd_ps(___x2_24_1, ___x0_1, ___x1_24);
                            ___x1_24 = _mm512_fmadd_ps(___x2_24_2, ___x0_2, ___x1_24);
                            ___x1_24 = _mm512_fmadd_ps(___x2_24_3, ___x0_3, ___x1_24);
                            ___x1_25 = _mm512_fmadd_ps(___x2_25_0, ___x0_0, ___x1_25);
                            ___x1_25 = _mm512_fmadd_ps(___x2_25_1, ___x0_1, ___x1_25);
                            ___x1_25 = _mm512_fmadd_ps(___x2_25_2, ___x0_2, ___x1_25);
                            ___x1_25 = _mm512_fmadd_ps(___x2_25_3, ___x0_3, ___x1_25);
                            ___x1_26 = _mm512_fmadd_ps(___x2_26_0, ___x0_0, ___x1_26);
                            ___x1_26 = _mm512_fmadd_ps(___x2_26_1, ___x0_1, ___x1_26);
                            ___x1_26 = _mm512_fmadd_ps(___x2_26_2, ___x0_2, ___x1_26);
                            ___x1_26 = _mm512_fmadd_ps(___x2_26_3, ___x0_3, ___x1_26);
                            ___x1_27 = _mm512_fmadd_ps(___x2_27_0, ___x0_0, ___x1_27);
                            ___x1_27 = _mm512_fmadd_ps(___x2_27_1, ___x0_1, ___x1_27);
                            ___x1_27 = _mm512_fmadd_ps(___x2_27_2, ___x0_2, ___x1_27);
                            ___x1_27 = _mm512_fmadd_ps(___x2_27_3, ___x0_3, ___x1_27);
                        }
                    }
                }
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 0 + 1)][0], ___x1_0);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1 + 1)][0], ___x1_1);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 2 + 1)][0], ___x1_2);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 3 + 1)][0], ___x1_3);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 4 + 1)][0], ___x1_4);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 5 + 1)][0], ___x1_5);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 6 + 1)][0], ___x1_6);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 7 + 1)][0], ___x1_7);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 8 + 1)][0], ___x1_8);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 9 + 1)][0], ___x1_9);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 10 + 1)][0], ___x1_10);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 11 + 1)][0], ___x1_11);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 12 + 1)][0], ___x1_12);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 13 + 1)][0], ___x1_13);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 14 + 1)][0], ___x1_14);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 15 + 1)][0], ___x1_15);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 16 + 1)][0], ___x1_16);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 17 + 1)][0], ___x1_17);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 18 + 1)][0], ___x1_18);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 19 + 1)][0], ___x1_19);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 20 + 1)][0], ___x1_20);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 21 + 1)][0], ___x1_21);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 22 + 1)][0], ___x1_22);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 23 + 1)][0], ___x1_23);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 24 + 1)][0], ___x1_24);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 25 + 1)][0], ___x1_25);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 26 + 1)][0], ___x1_26);
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 27 + 1)][0], ___x1_27);
            }
        }
    }
    for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
        for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 1) {
            for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                ensemble3value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] + ensemble3bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                ensemble4value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
            }
        }
    }
    ;
        }
      }
    );;
        }
      }
    );
    
    parallel_for(0, 8,
      [=](int low, int high) {
        for (int x0 = low; x0 < high; x0++) {
          for (int x1 = 0; x1 < 6; x1 ++) {
        for (int x2 = 0; x2 < 3; x2 ++) {
            for (int x3 = 0; x3 < 3; x3 ++) {
                transpose<SIMDWIDTH,SIMDWIDTH>(& ensemble5weights[x0][x1][x2][x3][0][0], & ensemble5weights_transposed[x0][x1][x2][x3][0][0]);
            }
        }
    }
        } 
      }
    );
    
    parallel_for(0,128 / 1,
      [=](int low, int high) {
        for (int tmp__neuron_index_0 = low; tmp__neuron_index_0 < high; tmp__neuron_index_0++) {
          int _neuron_index_0 = tmp__neuron_index_0 * 1;
          
    parallel_for(0,8 / 1,
      [=](int low, int high) {
        for (int tmp__neuron_index_1_outer = low; tmp__neuron_index_1_outer < high; tmp__neuron_index_1_outer++) {
          int _neuron_index_1_outer = tmp__neuron_index_1_outer * 1;
          for (int i_outer = 0; i_outer < 6; i_outer += 1) {
        for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
            int in_y = _neuron_index_2 * 1;
            int _input_offset_2 = in_y;
            for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 28) {
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
                int in_x_14 = (_neuron_index_3 + 14) * 1;
                int in_x_15 = (_neuron_index_3 + 15) * 1;
                int in_x_16 = (_neuron_index_3 + 16) * 1;
                int in_x_17 = (_neuron_index_3 + 17) * 1;
                int in_x_18 = (_neuron_index_3 + 18) * 1;
                int in_x_19 = (_neuron_index_3 + 19) * 1;
                int in_x_20 = (_neuron_index_3 + 20) * 1;
                int in_x_21 = (_neuron_index_3 + 21) * 1;
                int in_x_22 = (_neuron_index_3 + 22) * 1;
                int in_x_23 = (_neuron_index_3 + 23) * 1;
                int in_x_24 = (_neuron_index_3 + 24) * 1;
                int in_x_25 = (_neuron_index_3 + 25) * 1;
                int in_x_26 = (_neuron_index_3 + 26) * 1;
                int in_x_27 = (_neuron_index_3 + 27) * 1;
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
                int _input_offset_3_14 = in_x_14;
                int _input_offset_3_15 = in_x_15;
                int _input_offset_3_16 = in_x_16;
                int _input_offset_3_17 = in_x_17;
                int _input_offset_3_18 = in_x_18;
                int _input_offset_3_19 = in_x_19;
                int _input_offset_3_20 = in_x_20;
                int _input_offset_3_21 = in_x_21;
                int _input_offset_3_22 = in_x_22;
                int _input_offset_3_23 = in_x_23;
                int _input_offset_3_24 = in_x_24;
                int _input_offset_3_25 = in_x_25;
                int _input_offset_3_26 = in_x_26;
                int _input_offset_3_27 = in_x_27;
                __m512 ___x8_0 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 0)][0]);
                __m512 ___x8_1 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 1)][0]);
                __m512 ___x8_2 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 2)][0]);
                __m512 ___x8_3 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 3)][0]);
                __m512 ___x8_4 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 4)][0]);
                __m512 ___x8_5 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 5)][0]);
                __m512 ___x8_6 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 6)][0]);
                __m512 ___x8_7 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 7)][0]);
                __m512 ___x8_8 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 8)][0]);
                __m512 ___x8_9 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 9)][0]);
                __m512 ___x8_10 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 10)][0]);
                __m512 ___x8_11 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 11)][0]);
                __m512 ___x8_12 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 12)][0]);
                __m512 ___x8_13 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 13)][0]);
                __m512 ___x8_14 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 14)][0]);
                __m512 ___x8_15 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 15)][0]);
                __m512 ___x8_16 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 16)][0]);
                __m512 ___x8_17 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 17)][0]);
                __m512 ___x8_18 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 18)][0]);
                __m512 ___x8_19 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 19)][0]);
                __m512 ___x8_20 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 20)][0]);
                __m512 ___x8_21 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 21)][0]);
                __m512 ___x8_22 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 22)][0]);
                __m512 ___x8_23 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 23)][0]);
                __m512 ___x8_24 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 24)][0]);
                __m512 ___x8_25 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 25)][0]);
                __m512 ___x8_26 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 26)][0]);
                __m512 ___x8_27 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 27)][0]);
                for (int j = 0; j < 3; j += 1) {
                    for (int k = 0; k < 3; k += 1) {
                        for (int i_inner = 0; i_inner < 16; i_inner += 4) {
                            __m512 ___x6_0 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                            __m512 ___x6_1 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                            __m512 ___x6_2 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 2)][0]);
                            __m512 ___x6_3 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 3)][0]);
                            __m512 ___x7_0_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 0)]);
                            __m512 ___x7_0_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 1)]);
                            __m512 ___x7_0_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 2)]);
                            __m512 ___x7_0_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 3)]);
                            __m512 ___x7_1_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 0)]);
                            __m512 ___x7_1_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 1)]);
                            __m512 ___x7_1_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 2)]);
                            __m512 ___x7_1_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_1)][(i_inner + 3)]);
                            __m512 ___x7_2_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 0)]);
                            __m512 ___x7_2_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 1)]);
                            __m512 ___x7_2_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 2)]);
                            __m512 ___x7_2_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_2)][(i_inner + 3)]);
                            __m512 ___x7_3_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 0)]);
                            __m512 ___x7_3_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 1)]);
                            __m512 ___x7_3_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 2)]);
                            __m512 ___x7_3_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_3)][(i_inner + 3)]);
                            __m512 ___x7_4_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 0)]);
                            __m512 ___x7_4_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 1)]);
                            __m512 ___x7_4_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 2)]);
                            __m512 ___x7_4_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_4)][(i_inner + 3)]);
                            __m512 ___x7_5_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 0)]);
                            __m512 ___x7_5_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 1)]);
                            __m512 ___x7_5_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 2)]);
                            __m512 ___x7_5_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_5)][(i_inner + 3)]);
                            __m512 ___x7_6_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 0)]);
                            __m512 ___x7_6_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 1)]);
                            __m512 ___x7_6_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 2)]);
                            __m512 ___x7_6_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_6)][(i_inner + 3)]);
                            __m512 ___x7_7_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 0)]);
                            __m512 ___x7_7_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 1)]);
                            __m512 ___x7_7_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 2)]);
                            __m512 ___x7_7_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_7)][(i_inner + 3)]);
                            __m512 ___x7_8_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 0)]);
                            __m512 ___x7_8_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 1)]);
                            __m512 ___x7_8_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 2)]);
                            __m512 ___x7_8_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_8)][(i_inner + 3)]);
                            __m512 ___x7_9_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 0)]);
                            __m512 ___x7_9_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 1)]);
                            __m512 ___x7_9_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 2)]);
                            __m512 ___x7_9_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_9)][(i_inner + 3)]);
                            __m512 ___x7_10_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 0)]);
                            __m512 ___x7_10_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 1)]);
                            __m512 ___x7_10_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 2)]);
                            __m512 ___x7_10_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_10)][(i_inner + 3)]);
                            __m512 ___x7_11_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 0)]);
                            __m512 ___x7_11_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 1)]);
                            __m512 ___x7_11_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 2)]);
                            __m512 ___x7_11_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_11)][(i_inner + 3)]);
                            __m512 ___x7_12_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 0)]);
                            __m512 ___x7_12_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 1)]);
                            __m512 ___x7_12_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 2)]);
                            __m512 ___x7_12_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_12)][(i_inner + 3)]);
                            __m512 ___x7_13_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 0)]);
                            __m512 ___x7_13_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 1)]);
                            __m512 ___x7_13_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 2)]);
                            __m512 ___x7_13_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_13)][(i_inner + 3)]);
                            __m512 ___x7_14_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 0)]);
                            __m512 ___x7_14_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 1)]);
                            __m512 ___x7_14_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 2)]);
                            __m512 ___x7_14_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_14)][(i_inner + 3)]);
                            __m512 ___x7_15_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 0)]);
                            __m512 ___x7_15_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 1)]);
                            __m512 ___x7_15_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 2)]);
                            __m512 ___x7_15_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_15)][(i_inner + 3)]);
                            __m512 ___x7_16_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 0)]);
                            __m512 ___x7_16_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 1)]);
                            __m512 ___x7_16_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 2)]);
                            __m512 ___x7_16_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_16)][(i_inner + 3)]);
                            __m512 ___x7_17_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 0)]);
                            __m512 ___x7_17_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 1)]);
                            __m512 ___x7_17_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 2)]);
                            __m512 ___x7_17_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_17)][(i_inner + 3)]);
                            __m512 ___x7_18_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 0)]);
                            __m512 ___x7_18_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 1)]);
                            __m512 ___x7_18_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 2)]);
                            __m512 ___x7_18_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_18)][(i_inner + 3)]);
                            __m512 ___x7_19_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 0)]);
                            __m512 ___x7_19_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 1)]);
                            __m512 ___x7_19_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 2)]);
                            __m512 ___x7_19_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_19)][(i_inner + 3)]);
                            __m512 ___x7_20_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 0)]);
                            __m512 ___x7_20_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 1)]);
                            __m512 ___x7_20_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 2)]);
                            __m512 ___x7_20_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_20)][(i_inner + 3)]);
                            __m512 ___x7_21_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 0)]);
                            __m512 ___x7_21_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 1)]);
                            __m512 ___x7_21_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 2)]);
                            __m512 ___x7_21_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_21)][(i_inner + 3)]);
                            __m512 ___x7_22_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 0)]);
                            __m512 ___x7_22_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 1)]);
                            __m512 ___x7_22_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 2)]);
                            __m512 ___x7_22_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_22)][(i_inner + 3)]);
                            __m512 ___x7_23_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 0)]);
                            __m512 ___x7_23_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 1)]);
                            __m512 ___x7_23_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 2)]);
                            __m512 ___x7_23_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_23)][(i_inner + 3)]);
                            __m512 ___x7_24_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 0)]);
                            __m512 ___x7_24_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 1)]);
                            __m512 ___x7_24_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 2)]);
                            __m512 ___x7_24_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_24)][(i_inner + 3)]);
                            __m512 ___x7_25_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 0)]);
                            __m512 ___x7_25_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 1)]);
                            __m512 ___x7_25_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 2)]);
                            __m512 ___x7_25_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_25)][(i_inner + 3)]);
                            __m512 ___x7_26_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 0)]);
                            __m512 ___x7_26_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 1)]);
                            __m512 ___x7_26_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 2)]);
                            __m512 ___x7_26_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_26)][(i_inner + 3)]);
                            __m512 ___x7_27_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 0)]);
                            __m512 ___x7_27_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 1)]);
                            __m512 ___x7_27_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 2)]);
                            __m512 ___x7_27_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_27)][(i_inner + 3)]);
                            ___x8_0 = _mm512_fmadd_ps(___x7_0_0, ___x6_0, ___x8_0);
                            ___x8_0 = _mm512_fmadd_ps(___x7_0_1, ___x6_1, ___x8_0);
                            ___x8_0 = _mm512_fmadd_ps(___x7_0_2, ___x6_2, ___x8_0);
                            ___x8_0 = _mm512_fmadd_ps(___x7_0_3, ___x6_3, ___x8_0);
                            ___x8_1 = _mm512_fmadd_ps(___x7_1_0, ___x6_0, ___x8_1);
                            ___x8_1 = _mm512_fmadd_ps(___x7_1_1, ___x6_1, ___x8_1);
                            ___x8_1 = _mm512_fmadd_ps(___x7_1_2, ___x6_2, ___x8_1);
                            ___x8_1 = _mm512_fmadd_ps(___x7_1_3, ___x6_3, ___x8_1);
                            ___x8_2 = _mm512_fmadd_ps(___x7_2_0, ___x6_0, ___x8_2);
                            ___x8_2 = _mm512_fmadd_ps(___x7_2_1, ___x6_1, ___x8_2);
                            ___x8_2 = _mm512_fmadd_ps(___x7_2_2, ___x6_2, ___x8_2);
                            ___x8_2 = _mm512_fmadd_ps(___x7_2_3, ___x6_3, ___x8_2);
                            ___x8_3 = _mm512_fmadd_ps(___x7_3_0, ___x6_0, ___x8_3);
                            ___x8_3 = _mm512_fmadd_ps(___x7_3_1, ___x6_1, ___x8_3);
                            ___x8_3 = _mm512_fmadd_ps(___x7_3_2, ___x6_2, ___x8_3);
                            ___x8_3 = _mm512_fmadd_ps(___x7_3_3, ___x6_3, ___x8_3);
                            ___x8_4 = _mm512_fmadd_ps(___x7_4_0, ___x6_0, ___x8_4);
                            ___x8_4 = _mm512_fmadd_ps(___x7_4_1, ___x6_1, ___x8_4);
                            ___x8_4 = _mm512_fmadd_ps(___x7_4_2, ___x6_2, ___x8_4);
                            ___x8_4 = _mm512_fmadd_ps(___x7_4_3, ___x6_3, ___x8_4);
                            ___x8_5 = _mm512_fmadd_ps(___x7_5_0, ___x6_0, ___x8_5);
                            ___x8_5 = _mm512_fmadd_ps(___x7_5_1, ___x6_1, ___x8_5);
                            ___x8_5 = _mm512_fmadd_ps(___x7_5_2, ___x6_2, ___x8_5);
                            ___x8_5 = _mm512_fmadd_ps(___x7_5_3, ___x6_3, ___x8_5);
                            ___x8_6 = _mm512_fmadd_ps(___x7_6_0, ___x6_0, ___x8_6);
                            ___x8_6 = _mm512_fmadd_ps(___x7_6_1, ___x6_1, ___x8_6);
                            ___x8_6 = _mm512_fmadd_ps(___x7_6_2, ___x6_2, ___x8_6);
                            ___x8_6 = _mm512_fmadd_ps(___x7_6_3, ___x6_3, ___x8_6);
                            ___x8_7 = _mm512_fmadd_ps(___x7_7_0, ___x6_0, ___x8_7);
                            ___x8_7 = _mm512_fmadd_ps(___x7_7_1, ___x6_1, ___x8_7);
                            ___x8_7 = _mm512_fmadd_ps(___x7_7_2, ___x6_2, ___x8_7);
                            ___x8_7 = _mm512_fmadd_ps(___x7_7_3, ___x6_3, ___x8_7);
                            ___x8_8 = _mm512_fmadd_ps(___x7_8_0, ___x6_0, ___x8_8);
                            ___x8_8 = _mm512_fmadd_ps(___x7_8_1, ___x6_1, ___x8_8);
                            ___x8_8 = _mm512_fmadd_ps(___x7_8_2, ___x6_2, ___x8_8);
                            ___x8_8 = _mm512_fmadd_ps(___x7_8_3, ___x6_3, ___x8_8);
                            ___x8_9 = _mm512_fmadd_ps(___x7_9_0, ___x6_0, ___x8_9);
                            ___x8_9 = _mm512_fmadd_ps(___x7_9_1, ___x6_1, ___x8_9);
                            ___x8_9 = _mm512_fmadd_ps(___x7_9_2, ___x6_2, ___x8_9);
                            ___x8_9 = _mm512_fmadd_ps(___x7_9_3, ___x6_3, ___x8_9);
                            ___x8_10 = _mm512_fmadd_ps(___x7_10_0, ___x6_0, ___x8_10);
                            ___x8_10 = _mm512_fmadd_ps(___x7_10_1, ___x6_1, ___x8_10);
                            ___x8_10 = _mm512_fmadd_ps(___x7_10_2, ___x6_2, ___x8_10);
                            ___x8_10 = _mm512_fmadd_ps(___x7_10_3, ___x6_3, ___x8_10);
                            ___x8_11 = _mm512_fmadd_ps(___x7_11_0, ___x6_0, ___x8_11);
                            ___x8_11 = _mm512_fmadd_ps(___x7_11_1, ___x6_1, ___x8_11);
                            ___x8_11 = _mm512_fmadd_ps(___x7_11_2, ___x6_2, ___x8_11);
                            ___x8_11 = _mm512_fmadd_ps(___x7_11_3, ___x6_3, ___x8_11);
                            ___x8_12 = _mm512_fmadd_ps(___x7_12_0, ___x6_0, ___x8_12);
                            ___x8_12 = _mm512_fmadd_ps(___x7_12_1, ___x6_1, ___x8_12);
                            ___x8_12 = _mm512_fmadd_ps(___x7_12_2, ___x6_2, ___x8_12);
                            ___x8_12 = _mm512_fmadd_ps(___x7_12_3, ___x6_3, ___x8_12);
                            ___x8_13 = _mm512_fmadd_ps(___x7_13_0, ___x6_0, ___x8_13);
                            ___x8_13 = _mm512_fmadd_ps(___x7_13_1, ___x6_1, ___x8_13);
                            ___x8_13 = _mm512_fmadd_ps(___x7_13_2, ___x6_2, ___x8_13);
                            ___x8_13 = _mm512_fmadd_ps(___x7_13_3, ___x6_3, ___x8_13);
                            ___x8_14 = _mm512_fmadd_ps(___x7_14_0, ___x6_0, ___x8_14);
                            ___x8_14 = _mm512_fmadd_ps(___x7_14_1, ___x6_1, ___x8_14);
                            ___x8_14 = _mm512_fmadd_ps(___x7_14_2, ___x6_2, ___x8_14);
                            ___x8_14 = _mm512_fmadd_ps(___x7_14_3, ___x6_3, ___x8_14);
                            ___x8_15 = _mm512_fmadd_ps(___x7_15_0, ___x6_0, ___x8_15);
                            ___x8_15 = _mm512_fmadd_ps(___x7_15_1, ___x6_1, ___x8_15);
                            ___x8_15 = _mm512_fmadd_ps(___x7_15_2, ___x6_2, ___x8_15);
                            ___x8_15 = _mm512_fmadd_ps(___x7_15_3, ___x6_3, ___x8_15);
                            ___x8_16 = _mm512_fmadd_ps(___x7_16_0, ___x6_0, ___x8_16);
                            ___x8_16 = _mm512_fmadd_ps(___x7_16_1, ___x6_1, ___x8_16);
                            ___x8_16 = _mm512_fmadd_ps(___x7_16_2, ___x6_2, ___x8_16);
                            ___x8_16 = _mm512_fmadd_ps(___x7_16_3, ___x6_3, ___x8_16);
                            ___x8_17 = _mm512_fmadd_ps(___x7_17_0, ___x6_0, ___x8_17);
                            ___x8_17 = _mm512_fmadd_ps(___x7_17_1, ___x6_1, ___x8_17);
                            ___x8_17 = _mm512_fmadd_ps(___x7_17_2, ___x6_2, ___x8_17);
                            ___x8_17 = _mm512_fmadd_ps(___x7_17_3, ___x6_3, ___x8_17);
                            ___x8_18 = _mm512_fmadd_ps(___x7_18_0, ___x6_0, ___x8_18);
                            ___x8_18 = _mm512_fmadd_ps(___x7_18_1, ___x6_1, ___x8_18);
                            ___x8_18 = _mm512_fmadd_ps(___x7_18_2, ___x6_2, ___x8_18);
                            ___x8_18 = _mm512_fmadd_ps(___x7_18_3, ___x6_3, ___x8_18);
                            ___x8_19 = _mm512_fmadd_ps(___x7_19_0, ___x6_0, ___x8_19);
                            ___x8_19 = _mm512_fmadd_ps(___x7_19_1, ___x6_1, ___x8_19);
                            ___x8_19 = _mm512_fmadd_ps(___x7_19_2, ___x6_2, ___x8_19);
                            ___x8_19 = _mm512_fmadd_ps(___x7_19_3, ___x6_3, ___x8_19);
                            ___x8_20 = _mm512_fmadd_ps(___x7_20_0, ___x6_0, ___x8_20);
                            ___x8_20 = _mm512_fmadd_ps(___x7_20_1, ___x6_1, ___x8_20);
                            ___x8_20 = _mm512_fmadd_ps(___x7_20_2, ___x6_2, ___x8_20);
                            ___x8_20 = _mm512_fmadd_ps(___x7_20_3, ___x6_3, ___x8_20);
                            ___x8_21 = _mm512_fmadd_ps(___x7_21_0, ___x6_0, ___x8_21);
                            ___x8_21 = _mm512_fmadd_ps(___x7_21_1, ___x6_1, ___x8_21);
                            ___x8_21 = _mm512_fmadd_ps(___x7_21_2, ___x6_2, ___x8_21);
                            ___x8_21 = _mm512_fmadd_ps(___x7_21_3, ___x6_3, ___x8_21);
                            ___x8_22 = _mm512_fmadd_ps(___x7_22_0, ___x6_0, ___x8_22);
                            ___x8_22 = _mm512_fmadd_ps(___x7_22_1, ___x6_1, ___x8_22);
                            ___x8_22 = _mm512_fmadd_ps(___x7_22_2, ___x6_2, ___x8_22);
                            ___x8_22 = _mm512_fmadd_ps(___x7_22_3, ___x6_3, ___x8_22);
                            ___x8_23 = _mm512_fmadd_ps(___x7_23_0, ___x6_0, ___x8_23);
                            ___x8_23 = _mm512_fmadd_ps(___x7_23_1, ___x6_1, ___x8_23);
                            ___x8_23 = _mm512_fmadd_ps(___x7_23_2, ___x6_2, ___x8_23);
                            ___x8_23 = _mm512_fmadd_ps(___x7_23_3, ___x6_3, ___x8_23);
                            ___x8_24 = _mm512_fmadd_ps(___x7_24_0, ___x6_0, ___x8_24);
                            ___x8_24 = _mm512_fmadd_ps(___x7_24_1, ___x6_1, ___x8_24);
                            ___x8_24 = _mm512_fmadd_ps(___x7_24_2, ___x6_2, ___x8_24);
                            ___x8_24 = _mm512_fmadd_ps(___x7_24_3, ___x6_3, ___x8_24);
                            ___x8_25 = _mm512_fmadd_ps(___x7_25_0, ___x6_0, ___x8_25);
                            ___x8_25 = _mm512_fmadd_ps(___x7_25_1, ___x6_1, ___x8_25);
                            ___x8_25 = _mm512_fmadd_ps(___x7_25_2, ___x6_2, ___x8_25);
                            ___x8_25 = _mm512_fmadd_ps(___x7_25_3, ___x6_3, ___x8_25);
                            ___x8_26 = _mm512_fmadd_ps(___x7_26_0, ___x6_0, ___x8_26);
                            ___x8_26 = _mm512_fmadd_ps(___x7_26_1, ___x6_1, ___x8_26);
                            ___x8_26 = _mm512_fmadd_ps(___x7_26_2, ___x6_2, ___x8_26);
                            ___x8_26 = _mm512_fmadd_ps(___x7_26_3, ___x6_3, ___x8_26);
                            ___x8_27 = _mm512_fmadd_ps(___x7_27_0, ___x6_0, ___x8_27);
                            ___x8_27 = _mm512_fmadd_ps(___x7_27_1, ___x6_1, ___x8_27);
                            ___x8_27 = _mm512_fmadd_ps(___x7_27_2, ___x6_2, ___x8_27);
                            ___x8_27 = _mm512_fmadd_ps(___x7_27_3, ___x6_3, ___x8_27);
                        }
                    }
                }
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 0)][0], ___x8_0);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 1)][0], ___x8_1);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 2)][0], ___x8_2);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 3)][0], ___x8_3);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 4)][0], ___x8_4);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 5)][0], ___x8_5);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 6)][0], ___x8_6);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 7)][0], ___x8_7);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 8)][0], ___x8_8);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 9)][0], ___x8_9);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 10)][0], ___x8_10);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 11)][0], ___x8_11);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 12)][0], ___x8_12);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 13)][0], ___x8_13);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 14)][0], ___x8_14);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 15)][0], ___x8_15);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 16)][0], ___x8_16);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 17)][0], ___x8_17);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 18)][0], ___x8_18);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 19)][0], ___x8_19);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 20)][0], ___x8_20);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 21)][0], ___x8_21);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 22)][0], ___x8_22);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 23)][0], ___x8_23);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 24)][0], ___x8_24);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 25)][0], ___x8_25);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 26)][0], ___x8_26);
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 27)][0], ___x8_27);
            }
        }
    }
    for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
        for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 1) {
            for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                ensemble6value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble6inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] + ensemble6bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                ensemble7value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = MAX(ensemble7inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner], (float) 0.0);
            }
        }
    }
    for (int _neuron_index_2 = 0; _neuron_index_2 < 14; _neuron_index_2 += 1) {
        for (int _neuron_index_3 = 0; _neuron_index_3 < 14; _neuron_index_3 += 1) {
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
                        if (ensemble8inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j * 1 + _input_offset_2, 0), 27)][MIN(MAX(k * 1 + _input_offset_3, 0), 27)][_input_offset_1_inner] > max_value) {
                            max_value = ensemble8inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j * 1 + _input_offset_2, 0), 27)][MIN(MAX(k * 1 + _input_offset_3, 0), 27)][_input_offset_1_inner];
                            ensemble8mask_j[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = j;
                            ensemble8mask_k[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = k;
                        };
                    }
                }
                ensemble8value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = max_value;
            }
        }
    }
    ;
        }
      }
    );;
        }
      }
    );
};
