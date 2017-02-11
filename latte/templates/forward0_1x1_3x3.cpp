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
          
    parallel_for(0,6 / 1,
      [=](int low, int high) {
        for (int tmp__neuron_index_1_outer = low; tmp__neuron_index_1_outer < high; tmp__neuron_index_1_outer++) {
          int _neuron_index_1_outer = tmp__neuron_index_1_outer * 1;
          for (int i_outer = 0; i_outer < 1; i_outer += 1) {
        for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
            int in_y = _neuron_index_2 * 1;
            int _input_offset_2 = in_y;
            for (int _neuron_index_3 = 0; _neuron_index_3 < 3; _neuron_index_3 += 1) {
                int in_x_0 = (_neuron_index_3 + 0) * 1;
                int _input_offset_3_0 = in_x_0;
                __m512 ___x0_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 0 + 1)][0]);
                for (int j = 0; j < 1; j += 1) {
                    for (int k = 0; k < 1; k += 1) {
                        for (int i_inner = 0; i_inner < 16; i_inner += 4) {
                            __m512 ___x1_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 0)]);
                            __m512 ___x1_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 1)]);
                            __m512 ___x1_0_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 2)]);
                            __m512 ___x1_0_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 3)]);
                            __m512 ___x2_0 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                            __m512 ___x2_1 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                            __m512 ___x2_2 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 2)][0]);
                            __m512 ___x2_3 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 3)][0]);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_0, ___x2_0, ___x0_0);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_1, ___x2_1, ___x0_0);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_2, ___x2_2, ___x0_0);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_3, ___x2_3, ___x0_0);
                        }
                    }
                }
                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 0 + 1)][0], ___x0_0);
                ensemble3value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] + ensemble3bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                ensemble4value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
            for (int _neuron_index_3 = 3; _neuron_index_3 < 28; _neuron_index_3 += 1) {
                int in_x_0 = (_neuron_index_3 + 0) * 1;
                int _input_offset_3_0 = in_x_0;
                __m512 ___x0_0 = _mm512_load_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 0 + 1)][0]);
                for (int j = 0; j < 1; j += 1) {
                    for (int k = 0; k < 1; k += 1) {
                        for (int i_inner = 0; i_inner < 16; i_inner += 4) {
                            __m512 ___x1_0_0 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 0)]);
                            __m512 ___x1_0_1 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 1)]);
                            __m512 ___x1_0_2 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 2)]);
                            __m512 ___x1_0_3 = _mm512_set1_ps(ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 3)]);
                            __m512 ___x2_0 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                            __m512 ___x2_1 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                            __m512 ___x2_2 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 2)][0]);
                            __m512 ___x2_3 = _mm512_load_ps(& ensemble2weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 3)][0]);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_0, ___x2_0, ___x0_0);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_1, ___x2_1, ___x0_0);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_2, ___x2_2, ___x0_0);
                            ___x0_0 = _mm512_fmadd_ps(___x1_0_3, ___x2_3, ___x0_0);
                        }
                    }
                }

                _mm512_store_ps(& ensemble2value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 0 + 1)][0], ___x0_0);
                ensemble3value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = ensemble3inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] + ensemble3bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                ensemble4value[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = MAX(ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner], (float) 0.0);
                
                int in_x_0 = (_neuron_index_3 - 3) * 1;
                int _input_offset_3_0 = in_x_0;

                __m512 ___x6_0 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 -3 )][0]);
                for (int j = 0; j < 3; j += 1) {
                    for (int k = 0; k < 3; k += 1) {
                        for (int i_inner = 0; i_inner < 16; i_inner += 4) {
                            __m512 ___x7_0 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                            __m512 ___x7_1 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                            __m512 ___x7_2 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 2)][0]);
                            __m512 ___x7_3 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 3)][0]);
                            __m512 ___x8_0_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 0)]);
                            __m512 ___x8_0_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 1)]);
                            __m512 ___x8_0_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 2)]);
                            __m512 ___x8_0_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 3)]);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_0, ___x7_0, ___x6_0);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_1, ___x7_1, ___x6_0);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_2, ___x7_2, ___x6_0);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_3, ___x7_3, ___x6_0);
                        }
                    }
                }
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 -3)][0], ___x6_0);
                ensemble6value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner] = ensemble6inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner] + ensemble6bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                ensemble7value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner] = MAX(ensemble7inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner], (float) 0.0);
            }


             for (int _neuron_index_3 = 28; _neuron_index_3 < 31; _neuron_index_3 += 1) {
                int in_x_0 = (_neuron_index_3 - 3) * 1;
                int _input_offset_3_0 = in_x_0;
 
                __m512 ___x6_0 = _mm512_load_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 -3 )][0]);
                for (int j = 0; j < 3; j += 1) {
                    for (int k = 0; k < 3; k += 1) {
                        for (int i_inner = 0; i_inner < 16; i_inner += 4) {
                            __m512 ___x7_0 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 0)][0]);
                            __m512 ___x7_1 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 1)][0]);
                            __m512 ___x7_2 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 2)][0]);
                            __m512 ___x7_3 = _mm512_load_ps(& ensemble5weights_transposed[_neuron_index_1_outer][i_outer][j][k][(i_inner + 3)][0]);
                            __m512 ___x8_0_0 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 0)]);
                            __m512 ___x8_0_1 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 1)]);
                            __m512 ___x8_0_2 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 2)]);
                            __m512 ___x8_0_3 = _mm512_set1_ps(ensemble5inputs[_neuron_index_0][i_outer][(j * 1 + _input_offset_2)][(k * 1 + _input_offset_3_0)][(i_inner + 3)]);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_0, ___x7_0, ___x6_0);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_1, ___x7_1, ___x6_0);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_2, ___x7_2, ___x6_0);
                            ___x6_0 = _mm512_fmadd_ps(___x8_0_3, ___x7_3, ___x6_0);
                        }
                    }
                }
                _mm512_store_ps(& ensemble5value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 -3)][0], ___x6_0);
                ensemble6value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner] = ensemble6inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner] + ensemble6bias[_neuron_index_1_outer][0][_neuron_index_1_inner];
                ensemble7value[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner] = MAX(ensemble7inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3-3][_neuron_index_1_inner], (float) 0.0);


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
