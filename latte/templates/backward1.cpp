// <file: backward1.cpp>
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
void  backward1 (float* _, float* _ensemble2grad_inputsensemble2grad, float* _ensemble2grad_weightsensemble2grad, float* _ensemble2inputsensemble2grad, float* _ensemble2weightsensemble2grad, float* _ensemble3gradensemble2grad, float* _ensemble3grad_biasensemble2grad, float* _ensemble4gradensemble2grad, float* _ensemble4grad_inputsensemble2grad, float* _ensemble4inputsensemble2grad, float* _ensemble5gradensemble2grad, float* _ensemble5grad_inputsensemble2grad, long* _ensemble5mask_jensemble2grad, long* _ensemble5mask_k);
void  backward2 (long* , float* ensemble4inputsensemble5mask_k, float* ensemble4gradensemble5mask_k, float* ensemble4grad_inputsensemble5mask_k, float* ensemble5grad_inputsensemble5mask_k, long* ensemble5mask_jensemble5mask_k, float* ensemble5grad);
void  backward3 (float* , float* ensemble2inputsensemble2grad_weights, float* ensemble2gradensemble2grad_weights, float* ensemble3gradensemble2grad_weights, float* ensemble3grad_bias);
void  backward4 (float* , float* ensemble2gradensemble2weights, float* ensemble2grad_inputs);
void backward1(float* _ensemble2grad, float* _ensemble2grad_inputs, float* _ensemble2grad_weights, float* _ensemble2inputs, float* _ensemble2weights, float* _ensemble3grad, float* _ensemble3grad_bias, float* _ensemble4grad, float* _ensemble4grad_inputs, float* _ensemble4inputs, float* _ensemble5grad, float* _ensemble5grad_inputs, long* _ensemble5mask_j, long* _ensemble5mask_k) {
    long (* ensemble5mask_k)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_k;
    __assume_aligned(ensemble5mask_k, 64);
    long (* ensemble5mask_j)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_j;
    __assume_aligned(ensemble5mask_j, 64);
    float (* ensemble5grad_inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble5grad_inputs;
    __assume_aligned(ensemble5grad_inputs, 64);
    float (* ensemble5grad)[4][56][56][16] = (float (*)[4][56][56][16]) _ensemble5grad;
    __assume_aligned(ensemble5grad, 64);
    float (* ensemble4inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    float (* ensemble4grad_inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4grad_inputs;
    __assume_aligned(ensemble4grad_inputs, 64);
    float (* ensemble4grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4grad;
    __assume_aligned(ensemble4grad, 64);
    float (* ensemble3grad_bias)[1][16] = (float (*)[1][16]) _ensemble3grad_bias;
    __assume_aligned(ensemble3grad_bias, 64);
    float (* ensemble3grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble3grad;
    __assume_aligned(ensemble3grad, 64);
    float (* ensemble2weights)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2weights;
    __assume_aligned(ensemble2weights, 64);
    float (* ensemble2inputs)[1][230][230][16] = (float (*)[1][230][230][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    float (* ensemble2grad_weights)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2grad_weights;
    __assume_aligned(ensemble2grad_weights, 64);
    float (* ensemble2grad_inputs)[1][230][230][16] = (float (*)[1][230][230][16]) _ensemble2grad_inputs;
    __assume_aligned(ensemble2grad_inputs, 64);
    float (* ensemble2grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble2grad;
    __assume_aligned(ensemble2grad, 64);
    backward2(_ensemble5mask_k, _ensemble4inputs, _ensemble4grad, _ensemble4grad_inputs, _ensemble5grad_inputs, _ensemble5mask_j, _ensemble5grad);
    backward3(_ensemble2grad_weights, _ensemble2inputs, _ensemble2grad, _ensemble3grad, _ensemble3grad_bias);
    backward4(_ensemble2weights, _ensemble2grad, _ensemble2grad_inputs);
};
void backward2(long* _ensemble5mask_k, float* _ensemble4inputs, float* _ensemble4grad, float* _ensemble4grad_inputs, float* _ensemble5grad_inputs, long* _ensemble5mask_j, float* _ensemble5grad) {
    float (* ensemble5grad)[4][56][56][16] = (float (*)[4][56][56][16]) _ensemble5grad;
    __assume_aligned(ensemble5grad, 64);
    long (* ensemble5mask_j)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_j;
    __assume_aligned(ensemble5mask_j, 64);
    float (* ensemble5grad_inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble5grad_inputs;
    __assume_aligned(ensemble5grad_inputs, 64);
    float (* ensemble4grad_inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4grad_inputs;
    __assume_aligned(ensemble4grad_inputs, 64);
    float (* ensemble4grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4grad;
    __assume_aligned(ensemble4grad, 64);
    float (* ensemble4inputs)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    long (* ensemble5mask_k)[4][56][56][16] = (long (*)[4][56][56][16]) _ensemble5mask_k;
    __assume_aligned(ensemble5mask_k, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 4; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 56; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 56; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        int _input_offset_1_outer = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) / 16;
                        int _input_offset_1_inner = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) % 16;
                        int in_y = _neuron_index_2 * 2 - 0;
                        int _input_offset_2 = in_y;
                        int in_x = _neuron_index_3 * 2 - 0;
                        int _input_offset_3 = in_x;
                        long j = ensemble5mask_j[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        long k = ensemble5mask_k[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        ensemble5grad_inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j + _input_offset_2, 0), 111)][MIN(MAX(k + _input_offset_3, 0), 111)][_input_offset_1_inner] += ensemble5grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                    }
                }
            }
            for (int _neuron_index_2 = 0; _neuron_index_2 < 112; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 112; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        if (ensemble4inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] > 0.0) {
                            ensemble4grad_inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble4grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        } else {
                            ensemble4grad_inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = 0.0;
                        };
                    }
                }
            }
        }
    }
};
void backward3(float* _ensemble2grad_weights, float* _ensemble2inputs, float* _ensemble2grad, float* _ensemble3grad, float* _ensemble3grad_bias) {
    float (* ensemble3grad_bias)[1][16] = (float (*)[1][16]) _ensemble3grad_bias;
    __assume_aligned(ensemble3grad_bias, 64);
    float (* ensemble3grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble3grad;
    __assume_aligned(ensemble3grad, 64);
    float (* ensemble2grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble2grad;
    __assume_aligned(ensemble2grad, 64);
    float (* ensemble2inputs)[1][230][230][16] = (float (*)[1][230][230][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    float (* ensemble2grad_weights)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2grad_weights;
    __assume_aligned(ensemble2grad_weights, 64);
    #pragma omp parallel for
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 4; _neuron_index_1_outer += 1) {
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 112; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 112; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble3grad_bias[_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble3grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int i_outer = 0; i_outer < 1; i_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 112; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 2;
                    int _input_offset_2 = in_y;
                    for (int _neuron_index_3 = 0; _neuron_index_3 < 112; _neuron_index_3 += 28) {
                        int in_x_0 = (_neuron_index_3 + 0) * 2;
                        int in_x_1 = (_neuron_index_3 + 1) * 2;
                        int in_x_2 = (_neuron_index_3 + 2) * 2;
                        int in_x_3 = (_neuron_index_3 + 3) * 2;
                        int in_x_4 = (_neuron_index_3 + 4) * 2;
                        int in_x_5 = (_neuron_index_3 + 5) * 2;
                        int in_x_6 = (_neuron_index_3 + 6) * 2;
                        int in_x_7 = (_neuron_index_3 + 7) * 2;
                        int in_x_8 = (_neuron_index_3 + 8) * 2;
                        int in_x_9 = (_neuron_index_3 + 9) * 2;
                        int in_x_10 = (_neuron_index_3 + 10) * 2;
                        int in_x_11 = (_neuron_index_3 + 11) * 2;
                        int in_x_12 = (_neuron_index_3 + 12) * 2;
                        int in_x_13 = (_neuron_index_3 + 13) * 2;
                        int in_x_14 = (_neuron_index_3 + 14) * 2;
                        int in_x_15 = (_neuron_index_3 + 15) * 2;
                        int in_x_16 = (_neuron_index_3 + 16) * 2;
                        int in_x_17 = (_neuron_index_3 + 17) * 2;
                        int in_x_18 = (_neuron_index_3 + 18) * 2;
                        int in_x_19 = (_neuron_index_3 + 19) * 2;
                        int in_x_20 = (_neuron_index_3 + 20) * 2;
                        int in_x_21 = (_neuron_index_3 + 21) * 2;
                        int in_x_22 = (_neuron_index_3 + 22) * 2;
                        int in_x_23 = (_neuron_index_3 + 23) * 2;
                        int in_x_24 = (_neuron_index_3 + 24) * 2;
                        int in_x_25 = (_neuron_index_3 + 25) * 2;
                        int in_x_26 = (_neuron_index_3 + 26) * 2;
                        int in_x_27 = (_neuron_index_3 + 27) * 2;
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
                        for (int j = 0; j < 7; j += 1) {
                            for (int k = 0; k < 7; k += 1) {
                                __m512 ___x7_0 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 0) * 2)][0]);
                                __m512 ___x7_1 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 1) * 2)][0]);
                                __m512 ___x7_2 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 2) * 2)][0]);
                                __m512 ___x7_3 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 3) * 2)][0]);
                                __m512 ___x7_4 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 4) * 2)][0]);
                                __m512 ___x7_5 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 5) * 2)][0]);
                                __m512 ___x7_6 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 6) * 2)][0]);
                                __m512 ___x7_7 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 7) * 2)][0]);
                                __m512 ___x7_8 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 8) * 2)][0]);
                                __m512 ___x7_9 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 9) * 2)][0]);
                                __m512 ___x7_10 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 10) * 2)][0]);
                                __m512 ___x7_11 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 11) * 2)][0]);
                                __m512 ___x7_12 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 12) * 2)][0]);
                                __m512 ___x7_13 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 13) * 2)][0]);
                                __m512 ___x7_14 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 14) * 2)][0]);
                                __m512 ___x7_15 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 15) * 2)][0]);
                                __m512 ___x7_16 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 16) * 2)][0]);
                                __m512 ___x7_17 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 17) * 2)][0]);
                                __m512 ___x7_18 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 18) * 2)][0]);
                                __m512 ___x7_19 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 19) * 2)][0]);
                                __m512 ___x7_20 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 20) * 2)][0]);
                                __m512 ___x7_21 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 21) * 2)][0]);
                                __m512 ___x7_22 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 22) * 2)][0]);
                                __m512 ___x7_23 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 23) * 2)][0]);
                                __m512 ___x7_24 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 24) * 2)][0]);
                                __m512 ___x7_25 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 25) * 2)][0]);
                                __m512 ___x7_26 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 26) * 2)][0]);
                                __m512 ___x7_27 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + (_neuron_index_3 + 27) * 2)][0]);
                                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                    __m512 ___x6 = _mm512_load_ps(& ensemble2grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                    __m512 ___x8_0 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 0)][_neuron_index_1_inner]);
                                    __m512 ___x8_1 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 1)][_neuron_index_1_inner]);
                                    __m512 ___x8_2 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 2)][_neuron_index_1_inner]);
                                    __m512 ___x8_3 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 3)][_neuron_index_1_inner]);
                                    __m512 ___x8_4 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 4)][_neuron_index_1_inner]);
                                    __m512 ___x8_5 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 5)][_neuron_index_1_inner]);
                                    __m512 ___x8_6 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 6)][_neuron_index_1_inner]);
                                    __m512 ___x8_7 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 7)][_neuron_index_1_inner]);
                                    __m512 ___x8_8 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 8)][_neuron_index_1_inner]);
                                    __m512 ___x8_9 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 9)][_neuron_index_1_inner]);
                                    __m512 ___x8_10 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 10)][_neuron_index_1_inner]);
                                    __m512 ___x8_11 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 11)][_neuron_index_1_inner]);
                                    __m512 ___x8_12 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 12)][_neuron_index_1_inner]);
                                    __m512 ___x8_13 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 13)][_neuron_index_1_inner]);
                                    __m512 ___x8_14 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 14)][_neuron_index_1_inner]);
                                    __m512 ___x8_15 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 15)][_neuron_index_1_inner]);
                                    __m512 ___x8_16 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 16)][_neuron_index_1_inner]);
                                    __m512 ___x8_17 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 17)][_neuron_index_1_inner]);
                                    __m512 ___x8_18 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 18)][_neuron_index_1_inner]);
                                    __m512 ___x8_19 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 19)][_neuron_index_1_inner]);
                                    __m512 ___x8_20 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 20)][_neuron_index_1_inner]);
                                    __m512 ___x8_21 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 21)][_neuron_index_1_inner]);
                                    __m512 ___x8_22 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 22)][_neuron_index_1_inner]);
                                    __m512 ___x8_23 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 23)][_neuron_index_1_inner]);
                                    __m512 ___x8_24 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 24)][_neuron_index_1_inner]);
                                    __m512 ___x8_25 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 25)][_neuron_index_1_inner]);
                                    __m512 ___x8_26 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 26)][_neuron_index_1_inner]);
                                    __m512 ___x8_27 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 27)][_neuron_index_1_inner]);
                                    ___x6 = _mm512_fmadd_ps(___x8_0, ___x7_0, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_1, ___x7_1, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_2, ___x7_2, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_3, ___x7_3, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_4, ___x7_4, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_5, ___x7_5, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_6, ___x7_6, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_7, ___x7_7, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_8, ___x7_8, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_9, ___x7_9, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_10, ___x7_10, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_11, ___x7_11, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_12, ___x7_12, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_13, ___x7_13, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_14, ___x7_14, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_15, ___x7_15, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_16, ___x7_16, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_17, ___x7_17, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_18, ___x7_18, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_19, ___x7_19, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_20, ___x7_20, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_21, ___x7_21, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_22, ___x7_22, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_23, ___x7_23, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_24, ___x7_24, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_25, ___x7_25, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_26, ___x7_26, ___x6);
                                    ___x6 = _mm512_fmadd_ps(___x8_27, ___x7_27, ___x6);
                                    _mm512_store_ps(& ensemble2grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0], ___x6);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
};
void backward4(float* _ensemble2weights, float* _ensemble2grad, float* _ensemble2grad_inputs) {
    float (* ensemble2grad_inputs)[1][230][230][16] = (float (*)[1][230][230][16]) _ensemble2grad_inputs;
    __assume_aligned(ensemble2grad_inputs, 64);
    float (* ensemble2grad)[4][112][112][16] = (float (*)[4][112][112][16]) _ensemble2grad;
    __assume_aligned(ensemble2grad, 64);
    float (* ensemble2weights)[1][7][7][16][16] = (float (*)[1][7][7][16][16]) _ensemble2weights;
    __assume_aligned(ensemble2weights, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int i_outer = 0; i_outer < 1; i_outer += 1) {
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 4; _neuron_index_1_outer += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 112; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 2;
                    int _input_offset_2 = in_y;
                    for (int _neuron_index_3 = 0; _neuron_index_3 < 112; _neuron_index_3 += 1) {
                        int in_x = _neuron_index_3 * 2;
                        int _input_offset_3 = in_x;
                        for (int j = 0; j < 7; j += 1) {
                            for (int k = 0; k < 7; k += 1) {
                                __m512 ___x5 = _mm512_load_ps(& ensemble2grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + _neuron_index_3 * 2)][0]);
                                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                    __m512 ___x3 = _mm512_load_ps(& ensemble2weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                    __m512 ___x4 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner]);
                                    ___x5 = _mm512_fmadd_ps(___x4, ___x3, ___x5);
                                }
                                _mm512_store_ps(& ensemble2grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 2)][(k * 1 + _neuron_index_3 * 2)][0], ___x5);
                            }
                        }
                    }
                }
            }
        }
    }
};
