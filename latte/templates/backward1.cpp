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
void  backward1 (float* _, float* _ensemble10grad_inputsensemble10grad, float* _ensemble10grad_weightsensemble10grad, float* _ensemble10inputsensemble10grad, float* _ensemble10weightsensemble10grad, float* _ensemble11gradensemble10grad, float* _ensemble11grad_biasensemble10grad, float* _ensemble12gradensemble10grad, float* _ensemble12grad_inputsensemble10grad, float* _ensemble12inputsensemble10grad, float* _ensemble13gradensemble10grad, float* _ensemble13grad_inputsensemble10grad, float* _ensemble13grad_weightsensemble10grad, float* _ensemble13inputsensemble10grad, float* _ensemble13weightsensemble10grad, float* _ensemble14gradensemble10grad, float* _ensemble14grad_biasensemble10grad, float* _ensemble15gradensemble10grad, float* _ensemble15grad_inputsensemble10grad, float* _ensemble15inputsensemble10grad, float* _ensemble16gradensemble10grad, float* _ensemble16grad_inputsensemble10grad, float* _ensemble16grad_weightsensemble10grad, float* _ensemble16inputsensemble10grad, float* _ensemble16weightsensemble10grad, float* _ensemble17gradensemble10grad, float* _ensemble17grad_biasensemble10grad, float* _ensemble18gradensemble10grad, float* _ensemble18grad_inputsensemble10grad, float* _ensemble18inputsensemble10grad, float* _ensemble19gradensemble10grad, float* _ensemble19grad_inputsensemble10grad, long* _ensemble19mask_jensemble10grad, long* _ensemble19mask_kensemble10grad, float* _ensemble20gradensemble10grad, float* _ensemble20grad_inputsensemble10grad, float* _ensemble20grad_weightsensemble10grad, float* _ensemble20inputsensemble10grad, float* _ensemble20weightsensemble10grad, float* _ensemble21gradensemble10grad, float* _ensemble21grad_biasensemble10grad, float* _ensemble22gradensemble10grad, float* _ensemble22grad_inputsensemble10grad, float* _ensemble22grad_weightsensemble10grad, float* _ensemble22inputsensemble10grad, float* _ensemble22weightsensemble10grad, float* _ensemble23gradensemble10grad, float* _ensemble23grad_biasensemble10grad, float* _ensemble24gradensemble10grad, float* _ensemble24grad_inputsensemble10grad, float* _ensemble24grad_weightsensemble10grad, float* _ensemble24inputsensemble10grad, float* _ensemble24weightsensemble10grad, float* _ensemble25gradensemble10grad, float* _ensemble25grad_biasensemble10grad, float* _ensemble2gradensemble10grad, float* _ensemble2grad_weightsensemble10grad, float* _ensemble2inputsensemble10grad, float* _ensemble3gradensemble10grad, float* _ensemble3grad_biasensemble10grad, float* _ensemble4gradensemble10grad, float* _ensemble4grad_inputsensemble10grad, float* _ensemble4inputsensemble10grad, float* _ensemble5gradensemble10grad, float* _ensemble5grad_inputsensemble10grad, long* _ensemble5mask_jensemble10grad, long* _ensemble5mask_kensemble10grad, float* _ensemble6gradensemble10grad, float* _ensemble6grad_inputsensemble10grad, float* _ensemble6grad_weightsensemble10grad, float* _ensemble6inputsensemble10grad, float* _ensemble6weightsensemble10grad, float* _ensemble7gradensemble10grad, float* _ensemble7grad_biasensemble10grad, float* _ensemble8gradensemble10grad, float* _ensemble8grad_inputsensemble10grad, float* _ensemble8inputsensemble10grad, float* _ensemble9gradensemble10grad, float* _ensemble9grad_inputsensemble10grad, long* _ensemble9mask_jensemble10grad, long* _ensemble9mask_k);
void  backward2 (float* , float* ensemble25grad);
void  backward3 (float* , float* ensemble24gradensemble24inputs, float* ensemble24grad_weights);
void  backward4 (float* , float* ensemble24gradensemble24grad_inputs, float* ensemble24weights);
void  backward5 (float* , float* ensemble23grad_bias);
void  backward6 (float* , float* ensemble22inputsensemble22grad_weights, float* ensemble22grad);
void  backward7 (float* , float* ensemble22grad_inputsensemble22weights, float* ensemble22grad);
void  backward8 (float* , float* ensemble21grad);
void  backward9 (float* , float* ensemble20gradensemble20grad_weights, float* ensemble20inputs);
void  backward10 (float* , float* ensemble20grad_inputsensemble20grad, float* ensemble20weights);
void  backward11 (float* , long* ensemble19mask_kensemble19grad_inputs, float* ensemble18gradensemble19grad_inputs, float* ensemble18grad_inputsensemble19grad_inputs, float* ensemble19gradensemble19grad_inputs, long* ensemble19mask_jensemble19grad_inputs, float* ensemble18inputs);
void  backward12 (float* , float* ensemble16inputsensemble17grad_bias, float* ensemble16gradensemble17grad_bias, float* ensemble16grad_weightsensemble17grad_bias, float* ensemble17grad);
void  backward13 (float* , float* ensemble16weightsensemble15grad, float* ensemble15inputsensemble15grad, float* ensemble15grad_inputsensemble15grad, float* ensemble16gradensemble15grad, float* ensemble16grad_inputs);
void  backward14 (float* , float* ensemble13grad_weightsensemble14grad_bias, float* ensemble13inputsensemble14grad_bias, float* ensemble14gradensemble14grad_bias, float* ensemble13grad);
void  backward15 (float* , float* ensemble12inputsensemble12grad_inputs, float* ensemble13grad_inputsensemble12grad_inputs, float* ensemble13gradensemble12grad_inputs, float* ensemble12gradensemble12grad_inputs, float* ensemble13weights);
void  backward16 (float* , float* ensemble11gradensemble10inputs, float* ensemble11grad_biasensemble10inputs, float* ensemble10grad_weightsensemble10inputs, float* ensemble10grad);
void  backward17 (float* , float* ensemble10grad_inputsensemble10grad, float* ensemble10weightsensemble10grad, float* ensemble9grad_inputsensemble10grad, float* ensemble9gradensemble10grad, long* ensemble9mask_jensemble10grad, float* ensemble8grad_inputsensemble10grad, float* ensemble8inputsensemble10grad, float* ensemble8gradensemble10grad, long* ensemble9mask_k);
void  backward18 (float* , float* ensemble6inputsensemble6grad, float* ensemble6grad_weightsensemble6grad, float* ensemble7grad_biasensemble6grad, float* ensemble7grad);
void  backward19 (long* , float* ensemble5gradensemble5mask_k, long* ensemble5mask_jensemble5mask_k, float* ensemble4inputsensemble5mask_k, float* ensemble6weightsensemble5mask_k, float* ensemble4grad_inputsensemble5mask_k, float* ensemble6gradensemble5mask_k, float* ensemble4gradensemble5mask_k, float* ensemble5grad_inputsensemble5mask_k, float* ensemble6grad_inputs);
void  backward20 (float* , float* ensemble2gradensemble3grad_bias, float* ensemble2grad_weightsensemble3grad_bias, float* ensemble2inputsensemble3grad_bias, float* ensemble3grad);
void backward1(float* _ensemble10grad, float* _ensemble10grad_inputs, float* _ensemble10grad_weights, float* _ensemble10inputs, float* _ensemble10weights, float* _ensemble11grad, float* _ensemble11grad_bias, float* _ensemble12grad, float* _ensemble12grad_inputs, float* _ensemble12inputs, float* _ensemble13grad, float* _ensemble13grad_inputs, float* _ensemble13grad_weights, float* _ensemble13inputs, float* _ensemble13weights, float* _ensemble14grad, float* _ensemble14grad_bias, float* _ensemble15grad, float* _ensemble15grad_inputs, float* _ensemble15inputs, float* _ensemble16grad, float* _ensemble16grad_inputs, float* _ensemble16grad_weights, float* _ensemble16inputs, float* _ensemble16weights, float* _ensemble17grad, float* _ensemble17grad_bias, float* _ensemble18grad, float* _ensemble18grad_inputs, float* _ensemble18inputs, float* _ensemble19grad, float* _ensemble19grad_inputs, long* _ensemble19mask_j, long* _ensemble19mask_k, float* _ensemble20grad, float* _ensemble20grad_inputs, float* _ensemble20grad_weights, float* _ensemble20inputs, float* _ensemble20weights, float* _ensemble21grad, float* _ensemble21grad_bias, float* _ensemble22grad, float* _ensemble22grad_inputs, float* _ensemble22grad_weights, float* _ensemble22inputs, float* _ensemble22weights, float* _ensemble23grad, float* _ensemble23grad_bias, float* _ensemble24grad, float* _ensemble24grad_inputs, float* _ensemble24grad_weights, float* _ensemble24inputs, float* _ensemble24weights, float* _ensemble25grad, float* _ensemble25grad_bias, float* _ensemble2grad, float* _ensemble2grad_weights, float* _ensemble2inputs, float* _ensemble3grad, float* _ensemble3grad_bias, float* _ensemble4grad, float* _ensemble4grad_inputs, float* _ensemble4inputs, float* _ensemble5grad, float* _ensemble5grad_inputs, long* _ensemble5mask_j, long* _ensemble5mask_k, float* _ensemble6grad, float* _ensemble6grad_inputs, float* _ensemble6grad_weights, float* _ensemble6inputs, float* _ensemble6weights, float* _ensemble7grad, float* _ensemble7grad_bias, float* _ensemble8grad, float* _ensemble8grad_inputs, float* _ensemble8inputs, float* _ensemble9grad, float* _ensemble9grad_inputs, long* _ensemble9mask_j, long* _ensemble9mask_k) {
    long (* ensemble9mask_k)[12][15][15][16] = (long (*)[12][15][15][16]) _ensemble9mask_k;
    __assume_aligned(ensemble9mask_k, 64);
    long (* ensemble9mask_j)[12][15][15][16] = (long (*)[12][15][15][16]) _ensemble9mask_j;
    __assume_aligned(ensemble9mask_j, 64);
    float (* ensemble9grad_inputs)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble9grad_inputs;
    __assume_aligned(ensemble9grad_inputs, 64);
    float (* ensemble9grad)[12][19][19][16] = (float (*)[12][19][19][16]) _ensemble9grad;
    __assume_aligned(ensemble9grad, 64);
    float (* ensemble8inputs)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble8inputs;
    __assume_aligned(ensemble8inputs, 64);
    float (* ensemble8grad_inputs)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble8grad_inputs;
    __assume_aligned(ensemble8grad_inputs, 64);
    float (* ensemble8grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble8grad;
    __assume_aligned(ensemble8grad, 64);
    float (* ensemble7grad_bias)[1][16] = (float (*)[1][16]) _ensemble7grad_bias;
    __assume_aligned(ensemble7grad_bias, 64);
    float (* ensemble7grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble7grad;
    __assume_aligned(ensemble7grad, 64);
    float (* ensemble6weights)[4][5][5][16][16] = (float (*)[4][5][5][16][16]) _ensemble6weights;
    __assume_aligned(ensemble6weights, 64);
    float (* ensemble6inputs)[4][32][32][16] = (float (*)[4][32][32][16]) _ensemble6inputs;
    __assume_aligned(ensemble6inputs, 64);
    float (* ensemble6grad_weights)[4][5][5][16][16] = (float (*)[4][5][5][16][16]) _ensemble6grad_weights;
    __assume_aligned(ensemble6grad_weights, 64);
    float (* ensemble6grad_inputs)[4][32][32][16] = (float (*)[4][32][32][16]) _ensemble6grad_inputs;
    __assume_aligned(ensemble6grad_inputs, 64);
    float (* ensemble6grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble6grad;
    __assume_aligned(ensemble6grad, 64);
    long (* ensemble5mask_k)[4][28][28][16] = (long (*)[4][28][28][16]) _ensemble5mask_k;
    __assume_aligned(ensemble5mask_k, 64);
    long (* ensemble5mask_j)[4][28][28][16] = (long (*)[4][28][28][16]) _ensemble5mask_j;
    __assume_aligned(ensemble5mask_j, 64);
    float (* ensemble5grad_inputs)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble5grad_inputs;
    __assume_aligned(ensemble5grad_inputs, 64);
    float (* ensemble5grad)[4][32][32][16] = (float (*)[4][32][32][16]) _ensemble5grad;
    __assume_aligned(ensemble5grad, 64);
    float (* ensemble4inputs)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    float (* ensemble4grad_inputs)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble4grad_inputs;
    __assume_aligned(ensemble4grad_inputs, 64);
    float (* ensemble4grad)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble4grad;
    __assume_aligned(ensemble4grad, 64);
    float (* ensemble3grad_bias)[1][16] = (float (*)[1][16]) _ensemble3grad_bias;
    __assume_aligned(ensemble3grad_bias, 64);
    float (* ensemble3grad)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble3grad;
    __assume_aligned(ensemble3grad, 64);
    float (* ensemble2inputs)[1][228][228][16] = (float (*)[1][228][228][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    float (* ensemble2grad_weights)[1][11][11][16][16] = (float (*)[1][11][11][16][16]) _ensemble2grad_weights;
    __assume_aligned(ensemble2grad_weights, 64);
    float (* ensemble2grad)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble2grad;
    __assume_aligned(ensemble2grad, 64);
    float (* ensemble25grad_bias)[63][1][16] = (float (*)[63][1][16]) _ensemble25grad_bias;
    __assume_aligned(ensemble25grad_bias, 64);
    float (* ensemble25grad)[63][16] = (float (*)[63][16]) _ensemble25grad;
    __assume_aligned(ensemble25grad, 64);
    float (* ensemble24weights)[256][16][16] = (float (*)[256][16][16]) _ensemble24weights;
    __assume_aligned(ensemble24weights, 64);
    float (* ensemble24inputs)[256][16] = (float (*)[256][16]) _ensemble24inputs;
    __assume_aligned(ensemble24inputs, 64);
    float (* ensemble24grad_weights)[256][16][16] = (float (*)[256][16][16]) _ensemble24grad_weights;
    __assume_aligned(ensemble24grad_weights, 64);
    float (* ensemble24grad_inputs)[256][16] = (float (*)[256][16]) _ensemble24grad_inputs;
    __assume_aligned(ensemble24grad_inputs, 64);
    float (* ensemble24grad)[63][16] = (float (*)[63][16]) _ensemble24grad;
    __assume_aligned(ensemble24grad, 64);
    float (* ensemble23grad_bias)[256][1][16] = (float (*)[256][1][16]) _ensemble23grad_bias;
    __assume_aligned(ensemble23grad_bias, 64);
    float (* ensemble23grad)[256][16] = (float (*)[256][16]) _ensemble23grad;
    __assume_aligned(ensemble23grad, 64);
    float (* ensemble22weights)[256][16][16] = (float (*)[256][16][16]) _ensemble22weights;
    __assume_aligned(ensemble22weights, 64);
    float (* ensemble22inputs)[256][16] = (float (*)[256][16]) _ensemble22inputs;
    __assume_aligned(ensemble22inputs, 64);
    float (* ensemble22grad_weights)[256][16][16] = (float (*)[256][16][16]) _ensemble22grad_weights;
    __assume_aligned(ensemble22grad_weights, 64);
    float (* ensemble22grad_inputs)[256][16] = (float (*)[256][16]) _ensemble22grad_inputs;
    __assume_aligned(ensemble22grad_inputs, 64);
    float (* ensemble22grad)[256][16] = (float (*)[256][16]) _ensemble22grad;
    __assume_aligned(ensemble22grad, 64);
    float (* ensemble21grad_bias)[256][1][16] = (float (*)[256][1][16]) _ensemble21grad_bias;
    __assume_aligned(ensemble21grad_bias, 64);
    float (* ensemble21grad)[256][16] = (float (*)[256][16]) _ensemble21grad;
    __assume_aligned(ensemble21grad, 64);
    float (* ensemble20weights)[16][9][9][16][16] = (float (*)[16][9][9][16][16]) _ensemble20weights;
    __assume_aligned(ensemble20weights, 64);
    float (* ensemble20inputs)[16][9][9][16] = (float (*)[16][9][9][16]) _ensemble20inputs;
    __assume_aligned(ensemble20inputs, 64);
    float (* ensemble20grad_weights)[16][9][9][16][16] = (float (*)[16][9][9][16][16]) _ensemble20grad_weights;
    __assume_aligned(ensemble20grad_weights, 64);
    float (* ensemble20grad_inputs)[16][9][9][16] = (float (*)[16][9][9][16]) _ensemble20grad_inputs;
    __assume_aligned(ensemble20grad_inputs, 64);
    float (* ensemble20grad)[256][16] = (float (*)[256][16]) _ensemble20grad;
    __assume_aligned(ensemble20grad, 64);
    long (* ensemble19mask_k)[16][9][9][16] = (long (*)[16][9][9][16]) _ensemble19mask_k;
    __assume_aligned(ensemble19mask_k, 64);
    long (* ensemble19mask_j)[16][9][9][16] = (long (*)[16][9][9][16]) _ensemble19mask_j;
    __assume_aligned(ensemble19mask_j, 64);
    float (* ensemble19grad_inputs)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble19grad_inputs;
    __assume_aligned(ensemble19grad_inputs, 64);
    float (* ensemble19grad)[16][9][9][16] = (float (*)[16][9][9][16]) _ensemble19grad;
    __assume_aligned(ensemble19grad, 64);
    float (* ensemble18inputs)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble18inputs;
    __assume_aligned(ensemble18inputs, 64);
    float (* ensemble18grad_inputs)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble18grad_inputs;
    __assume_aligned(ensemble18grad_inputs, 64);
    float (* ensemble18grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble18grad;
    __assume_aligned(ensemble18grad, 64);
    float (* ensemble17grad_bias)[1][16] = (float (*)[1][16]) _ensemble17grad_bias;
    __assume_aligned(ensemble17grad_bias, 64);
    float (* ensemble17grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble17grad;
    __assume_aligned(ensemble17grad, 64);
    float (* ensemble16weights)[16][3][3][16][16] = (float (*)[16][3][3][16][16]) _ensemble16weights;
    __assume_aligned(ensemble16weights, 64);
    float (* ensemble16inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble16inputs;
    __assume_aligned(ensemble16inputs, 64);
    float (* ensemble16grad_weights)[16][3][3][16][16] = (float (*)[16][3][3][16][16]) _ensemble16grad_weights;
    __assume_aligned(ensemble16grad_weights, 64);
    float (* ensemble16grad_inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble16grad_inputs;
    __assume_aligned(ensemble16grad_inputs, 64);
    float (* ensemble16grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble16grad;
    __assume_aligned(ensemble16grad, 64);
    float (* ensemble15inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble15inputs;
    __assume_aligned(ensemble15inputs, 64);
    float (* ensemble15grad_inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble15grad_inputs;
    __assume_aligned(ensemble15grad_inputs, 64);
    float (* ensemble15grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble15grad;
    __assume_aligned(ensemble15grad, 64);
    float (* ensemble14grad_bias)[1][16] = (float (*)[1][16]) _ensemble14grad_bias;
    __assume_aligned(ensemble14grad_bias, 64);
    float (* ensemble14grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble14grad;
    __assume_aligned(ensemble14grad, 64);
    float (* ensemble13weights)[24][3][3][16][16] = (float (*)[24][3][3][16][16]) _ensemble13weights;
    __assume_aligned(ensemble13weights, 64);
    float (* ensemble13inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble13inputs;
    __assume_aligned(ensemble13inputs, 64);
    float (* ensemble13grad_weights)[24][3][3][16][16] = (float (*)[24][3][3][16][16]) _ensemble13grad_weights;
    __assume_aligned(ensemble13grad_weights, 64);
    float (* ensemble13grad_inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble13grad_inputs;
    __assume_aligned(ensemble13grad_inputs, 64);
    float (* ensemble13grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble13grad;
    __assume_aligned(ensemble13grad, 64);
    float (* ensemble12inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble12inputs;
    __assume_aligned(ensemble12inputs, 64);
    float (* ensemble12grad_inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble12grad_inputs;
    __assume_aligned(ensemble12grad_inputs, 64);
    float (* ensemble12grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble12grad;
    __assume_aligned(ensemble12grad, 64);
    float (* ensemble11grad_bias)[1][16] = (float (*)[1][16]) _ensemble11grad_bias;
    __assume_aligned(ensemble11grad_bias, 64);
    float (* ensemble11grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble11grad;
    __assume_aligned(ensemble11grad, 64);
    float (* ensemble10weights)[12][3][3][16][16] = (float (*)[12][3][3][16][16]) _ensemble10weights;
    __assume_aligned(ensemble10weights, 64);
    float (* ensemble10inputs)[12][19][19][16] = (float (*)[12][19][19][16]) _ensemble10inputs;
    __assume_aligned(ensemble10inputs, 64);
    float (* ensemble10grad_weights)[12][3][3][16][16] = (float (*)[12][3][3][16][16]) _ensemble10grad_weights;
    __assume_aligned(ensemble10grad_weights, 64);
    float (* ensemble10grad_inputs)[12][19][19][16] = (float (*)[12][19][19][16]) _ensemble10grad_inputs;
    __assume_aligned(ensemble10grad_inputs, 64);
    float (* ensemble10grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble10grad;
    __assume_aligned(ensemble10grad, 64);
    backward2(_ensemble25grad_bias, _ensemble25grad);
    backward3(_ensemble24inputs, _ensemble24grad, _ensemble24grad_weights);
    backward4(_ensemble24grad_inputs, _ensemble24grad, _ensemble24weights);
    backward5(_ensemble23grad, _ensemble23grad_bias);
    backward6(_ensemble22grad_weights, _ensemble22inputs, _ensemble22grad);
    backward7(_ensemble22weights, _ensemble22grad_inputs, _ensemble22grad);
    backward8(_ensemble21grad_bias, _ensemble21grad);
    backward9(_ensemble20grad_weights, _ensemble20grad, _ensemble20inputs);
    backward10(_ensemble20grad, _ensemble20grad_inputs, _ensemble20weights);
    backward11(_ensemble19grad_inputs, _ensemble19mask_k, _ensemble18grad, _ensemble18grad_inputs, _ensemble19grad, _ensemble19mask_j, _ensemble18inputs);
    backward12(_ensemble17grad_bias, _ensemble16inputs, _ensemble16grad, _ensemble16grad_weights, _ensemble17grad);
    backward13(_ensemble15grad, _ensemble16weights, _ensemble15inputs, _ensemble15grad_inputs, _ensemble16grad, _ensemble16grad_inputs);
    backward14(_ensemble14grad_bias, _ensemble13grad_weights, _ensemble13inputs, _ensemble14grad, _ensemble13grad);
    backward15(_ensemble12grad_inputs, _ensemble12inputs, _ensemble13grad_inputs, _ensemble13grad, _ensemble12grad, _ensemble13weights);
    backward16(_ensemble10inputs, _ensemble11grad, _ensemble11grad_bias, _ensemble10grad_weights, _ensemble10grad);
    backward17(_ensemble10grad, _ensemble10grad_inputs, _ensemble10weights, _ensemble9grad_inputs, _ensemble9grad, _ensemble9mask_j, _ensemble8grad_inputs, _ensemble8inputs, _ensemble8grad, _ensemble9mask_k);
    backward18(_ensemble6grad, _ensemble6inputs, _ensemble6grad_weights, _ensemble7grad_bias, _ensemble7grad);
    backward19(_ensemble5mask_k, _ensemble5grad, _ensemble5mask_j, _ensemble4inputs, _ensemble6weights, _ensemble4grad_inputs, _ensemble6grad, _ensemble4grad, _ensemble5grad_inputs, _ensemble6grad_inputs);
    backward20(_ensemble3grad_bias, _ensemble2grad, _ensemble2grad_weights, _ensemble2inputs, _ensemble3grad);
};
void backward2(float* _ensemble25grad_bias, float* _ensemble25grad) {
    float (* ensemble25grad)[63][16] = (float (*)[63][16]) _ensemble25grad;
    __assume_aligned(ensemble25grad, 64);
    float (* ensemble25grad_bias)[63][1][16] = (float (*)[63][1][16]) _ensemble25grad_bias;
    __assume_aligned(ensemble25grad_bias, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 63; _neuron_index_1_outer += 1) {
            for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                ensemble25grad_bias[_neuron_index_0][_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble25grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_1_inner];
            }
        }
    }
    
    #pragma omp parallel for simd
    for (int x = 0; x < 1008; ++x) {
      float sum = _ensemble25grad_bias[x];
      #pragma unroll
      for (int i = 1; i < 128; ++ i) {
        sum += _ensemble25grad_bias[i * 1008 + x];
      }
      _ensemble25grad_bias[x] = sum;
    }
};
void backward3(float* _ensemble24inputs, float* _ensemble24grad, float* _ensemble24grad_weights) {
    float (* ensemble24grad_weights)[256][16][16] = (float (*)[256][16][16]) _ensemble24grad_weights;
    __assume_aligned(ensemble24grad_weights, 64);
    float (* ensemble24grad)[63][16] = (float (*)[63][16]) _ensemble24grad;
    __assume_aligned(ensemble24grad, 64);
    float (* ensemble24inputs)[256][16] = (float (*)[256][16]) _ensemble24inputs;
    __assume_aligned(ensemble24inputs, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 63; _neuron_index_1_outer += 1) {
        for (int __unique_loopvar0_outer = 0; __unique_loopvar0_outer < 256; __unique_loopvar0_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 16) {
                __m512 ___x67_0 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][0]);
                __m512 ___x67_1 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][0]);
                __m512 ___x67_2 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][0]);
                __m512 ___x67_3 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][0]);
                __m512 ___x67_4 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][0]);
                __m512 ___x67_5 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][0]);
                __m512 ___x67_6 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][0]);
                __m512 ___x67_7 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][0]);
                __m512 ___x67_8 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][0]);
                __m512 ___x67_9 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][0]);
                __m512 ___x67_10 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][0]);
                __m512 ___x67_11 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][0]);
                __m512 ___x67_12 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][0]);
                __m512 ___x67_13 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][0]);
                __m512 ___x67_14 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][0]);
                __m512 ___x67_15 = _mm512_load_ps(& ensemble24inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][0]);
                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                    __m512 ___x66 = _mm512_load_ps(& ensemble24grad_weights[_neuron_index_1_outer][__unique_loopvar0_outer][_neuron_index_1_inner][0]);
                    __m512 ___x68_0 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 0)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_1 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 1)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_2 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 2)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_3 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 3)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_4 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 4)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_5 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 5)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_6 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 6)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_7 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 7)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_8 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 8)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_9 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 9)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_10 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 10)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_11 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 11)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_12 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 12)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_13 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 13)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_14 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 14)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x68_15 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 15)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    ___x66 = _mm512_fmadd_ps(___x68_0, ___x67_0, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_1, ___x67_1, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_2, ___x67_2, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_3, ___x67_3, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_4, ___x67_4, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_5, ___x67_5, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_6, ___x67_6, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_7, ___x67_7, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_8, ___x67_8, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_9, ___x67_9, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_10, ___x67_10, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_11, ___x67_11, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_12, ___x67_12, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_13, ___x67_13, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_14, ___x67_14, ___x66);
                    ___x66 = _mm512_fmadd_ps(___x68_15, ___x67_15, ___x66);
                    _mm512_store_ps(& ensemble24grad_weights[_neuron_index_1_outer][__unique_loopvar0_outer][_neuron_index_1_inner][0], ___x66);
                }
            }
        }
    }
};
void backward4(float* _ensemble24grad_inputs, float* _ensemble24grad, float* _ensemble24weights) {
    float (* ensemble24weights)[256][16][16] = (float (*)[256][16][16]) _ensemble24weights;
    __assume_aligned(ensemble24weights, 64);
    float (* ensemble24grad)[63][16] = (float (*)[63][16]) _ensemble24grad;
    __assume_aligned(ensemble24grad, 64);
    float (* ensemble24grad_inputs)[256][16] = (float (*)[256][16]) _ensemble24grad_inputs;
    __assume_aligned(ensemble24grad_inputs, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 16) {
        for (int __unique_loopvar0_outer = 0; __unique_loopvar0_outer < 256; __unique_loopvar0_outer += 1) {
            __m512 ___x65_0 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][0]);
            __m512 ___x65_1 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][0]);
            __m512 ___x65_2 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][0]);
            __m512 ___x65_3 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][0]);
            __m512 ___x65_4 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][0]);
            __m512 ___x65_5 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][0]);
            __m512 ___x65_6 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][0]);
            __m512 ___x65_7 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][0]);
            __m512 ___x65_8 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][0]);
            __m512 ___x65_9 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][0]);
            __m512 ___x65_10 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][0]);
            __m512 ___x65_11 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][0]);
            __m512 ___x65_12 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][0]);
            __m512 ___x65_13 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][0]);
            __m512 ___x65_14 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][0]);
            __m512 ___x65_15 = _mm512_load_ps(& ensemble24grad_inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][0]);
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 63; _neuron_index_1_outer += 1) {
                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                    __m512 ___x63 = _mm512_load_ps(& ensemble24weights[_neuron_index_1_outer][__unique_loopvar0_outer][_neuron_index_1_inner][0]);
                    __m512 ___x64_0 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 0)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_1 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 1)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_2 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 2)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_3 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 3)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_4 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 4)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_5 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 5)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_6 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 6)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_7 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 7)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_8 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 8)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_9 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 9)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_10 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 10)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_11 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 11)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_12 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 12)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_13 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 13)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_14 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 14)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x64_15 = _mm512_set1_ps(ensemble24grad[(_neuron_index_0 + 15)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    ___x65_0 = _mm512_fmadd_ps(___x64_0, ___x63, ___x65_0);
                    ___x65_1 = _mm512_fmadd_ps(___x64_1, ___x63, ___x65_1);
                    ___x65_2 = _mm512_fmadd_ps(___x64_2, ___x63, ___x65_2);
                    ___x65_3 = _mm512_fmadd_ps(___x64_3, ___x63, ___x65_3);
                    ___x65_4 = _mm512_fmadd_ps(___x64_4, ___x63, ___x65_4);
                    ___x65_5 = _mm512_fmadd_ps(___x64_5, ___x63, ___x65_5);
                    ___x65_6 = _mm512_fmadd_ps(___x64_6, ___x63, ___x65_6);
                    ___x65_7 = _mm512_fmadd_ps(___x64_7, ___x63, ___x65_7);
                    ___x65_8 = _mm512_fmadd_ps(___x64_8, ___x63, ___x65_8);
                    ___x65_9 = _mm512_fmadd_ps(___x64_9, ___x63, ___x65_9);
                    ___x65_10 = _mm512_fmadd_ps(___x64_10, ___x63, ___x65_10);
                    ___x65_11 = _mm512_fmadd_ps(___x64_11, ___x63, ___x65_11);
                    ___x65_12 = _mm512_fmadd_ps(___x64_12, ___x63, ___x65_12);
                    ___x65_13 = _mm512_fmadd_ps(___x64_13, ___x63, ___x65_13);
                    ___x65_14 = _mm512_fmadd_ps(___x64_14, ___x63, ___x65_14);
                    ___x65_15 = _mm512_fmadd_ps(___x64_15, ___x63, ___x65_15);
                }
            }
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][0], ___x65_0);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][0], ___x65_1);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][0], ___x65_2);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][0], ___x65_3);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][0], ___x65_4);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][0], ___x65_5);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][0], ___x65_6);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][0], ___x65_7);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][0], ___x65_8);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][0], ___x65_9);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][0], ___x65_10);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][0], ___x65_11);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][0], ___x65_12);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][0], ___x65_13);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][0], ___x65_14);
            _mm512_store_ps(& ensemble24grad_inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][0], ___x65_15);
        }
    }
};
void backward5(float* _ensemble23grad, float* _ensemble23grad_bias) {
    float (* ensemble23grad_bias)[256][1][16] = (float (*)[256][1][16]) _ensemble23grad_bias;
    __assume_aligned(ensemble23grad_bias, 64);
    float (* ensemble23grad)[256][16] = (float (*)[256][16]) _ensemble23grad;
    __assume_aligned(ensemble23grad, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 256; _neuron_index_1_outer += 1) {
            for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                ensemble23grad_bias[_neuron_index_0][_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble23grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_1_inner];
            }
        }
    }
    
    #pragma omp parallel for simd
    for (int x = 0; x < 4096; ++x) {
      float sum = _ensemble23grad_bias[x];
      #pragma unroll
      for (int i = 1; i < 128; ++ i) {
        sum += _ensemble23grad_bias[i * 4096 + x];
      }
      _ensemble23grad_bias[x] = sum;
    }
};
void backward6(float* _ensemble22grad_weights, float* _ensemble22inputs, float* _ensemble22grad) {
    float (* ensemble22grad)[256][16] = (float (*)[256][16]) _ensemble22grad;
    __assume_aligned(ensemble22grad, 64);
    float (* ensemble22inputs)[256][16] = (float (*)[256][16]) _ensemble22inputs;
    __assume_aligned(ensemble22inputs, 64);
    float (* ensemble22grad_weights)[256][16][16] = (float (*)[256][16][16]) _ensemble22grad_weights;
    __assume_aligned(ensemble22grad_weights, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 256; _neuron_index_1_outer += 1) {
        for (int __unique_loopvar0_outer = 0; __unique_loopvar0_outer < 256; __unique_loopvar0_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 16) {
                __m512 ___x59_0 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][0]);
                __m512 ___x59_1 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][0]);
                __m512 ___x59_2 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][0]);
                __m512 ___x59_3 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][0]);
                __m512 ___x59_4 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][0]);
                __m512 ___x59_5 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][0]);
                __m512 ___x59_6 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][0]);
                __m512 ___x59_7 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][0]);
                __m512 ___x59_8 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][0]);
                __m512 ___x59_9 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][0]);
                __m512 ___x59_10 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][0]);
                __m512 ___x59_11 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][0]);
                __m512 ___x59_12 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][0]);
                __m512 ___x59_13 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][0]);
                __m512 ___x59_14 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][0]);
                __m512 ___x59_15 = _mm512_load_ps(& ensemble22inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][0]);
                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                    __m512 ___x57 = _mm512_load_ps(& ensemble22grad_weights[_neuron_index_1_outer][__unique_loopvar0_outer][_neuron_index_1_inner][0]);
                    __m512 ___x58_0 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 0)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_1 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 1)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_2 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 2)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_3 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 3)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_4 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 4)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_5 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 5)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_6 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 6)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_7 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 7)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_8 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 8)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_9 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 9)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_10 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 10)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_11 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 11)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_12 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 12)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_13 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 13)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_14 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 14)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x58_15 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 15)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    ___x57 = _mm512_fmadd_ps(___x58_0, ___x59_0, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_1, ___x59_1, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_2, ___x59_2, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_3, ___x59_3, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_4, ___x59_4, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_5, ___x59_5, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_6, ___x59_6, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_7, ___x59_7, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_8, ___x59_8, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_9, ___x59_9, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_10, ___x59_10, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_11, ___x59_11, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_12, ___x59_12, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_13, ___x59_13, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_14, ___x59_14, ___x57);
                    ___x57 = _mm512_fmadd_ps(___x58_15, ___x59_15, ___x57);
                    _mm512_store_ps(& ensemble22grad_weights[_neuron_index_1_outer][__unique_loopvar0_outer][_neuron_index_1_inner][0], ___x57);
                }
            }
        }
    }
};
void backward7(float* _ensemble22weights, float* _ensemble22grad_inputs, float* _ensemble22grad) {
    float (* ensemble22grad)[256][16] = (float (*)[256][16]) _ensemble22grad;
    __assume_aligned(ensemble22grad, 64);
    float (* ensemble22grad_inputs)[256][16] = (float (*)[256][16]) _ensemble22grad_inputs;
    __assume_aligned(ensemble22grad_inputs, 64);
    float (* ensemble22weights)[256][16][16] = (float (*)[256][16][16]) _ensemble22weights;
    __assume_aligned(ensemble22weights, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 16) {
        for (int __unique_loopvar0_outer = 0; __unique_loopvar0_outer < 256; __unique_loopvar0_outer += 1) {
            __m512 ___x54_0 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][0]);
            __m512 ___x54_1 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][0]);
            __m512 ___x54_2 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][0]);
            __m512 ___x54_3 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][0]);
            __m512 ___x54_4 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][0]);
            __m512 ___x54_5 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][0]);
            __m512 ___x54_6 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][0]);
            __m512 ___x54_7 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][0]);
            __m512 ___x54_8 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][0]);
            __m512 ___x54_9 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][0]);
            __m512 ___x54_10 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][0]);
            __m512 ___x54_11 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][0]);
            __m512 ___x54_12 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][0]);
            __m512 ___x54_13 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][0]);
            __m512 ___x54_14 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][0]);
            __m512 ___x54_15 = _mm512_load_ps(& ensemble22grad_inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][0]);
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 256; _neuron_index_1_outer += 1) {
                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                    __m512 ___x55_0 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 0)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_1 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 1)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_2 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 2)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_3 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 3)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_4 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 4)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_5 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 5)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_6 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 6)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_7 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 7)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_8 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 8)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_9 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 9)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_10 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 10)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_11 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 11)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_12 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 12)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_13 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 13)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_14 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 14)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x55_15 = _mm512_set1_ps(ensemble22grad[(_neuron_index_0 + 15)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x56 = _mm512_load_ps(& ensemble22weights[_neuron_index_1_outer][__unique_loopvar0_outer][_neuron_index_1_inner][0]);
                    ___x54_0 = _mm512_fmadd_ps(___x55_0, ___x56, ___x54_0);
                    ___x54_1 = _mm512_fmadd_ps(___x55_1, ___x56, ___x54_1);
                    ___x54_2 = _mm512_fmadd_ps(___x55_2, ___x56, ___x54_2);
                    ___x54_3 = _mm512_fmadd_ps(___x55_3, ___x56, ___x54_3);
                    ___x54_4 = _mm512_fmadd_ps(___x55_4, ___x56, ___x54_4);
                    ___x54_5 = _mm512_fmadd_ps(___x55_5, ___x56, ___x54_5);
                    ___x54_6 = _mm512_fmadd_ps(___x55_6, ___x56, ___x54_6);
                    ___x54_7 = _mm512_fmadd_ps(___x55_7, ___x56, ___x54_7);
                    ___x54_8 = _mm512_fmadd_ps(___x55_8, ___x56, ___x54_8);
                    ___x54_9 = _mm512_fmadd_ps(___x55_9, ___x56, ___x54_9);
                    ___x54_10 = _mm512_fmadd_ps(___x55_10, ___x56, ___x54_10);
                    ___x54_11 = _mm512_fmadd_ps(___x55_11, ___x56, ___x54_11);
                    ___x54_12 = _mm512_fmadd_ps(___x55_12, ___x56, ___x54_12);
                    ___x54_13 = _mm512_fmadd_ps(___x55_13, ___x56, ___x54_13);
                    ___x54_14 = _mm512_fmadd_ps(___x55_14, ___x56, ___x54_14);
                    ___x54_15 = _mm512_fmadd_ps(___x55_15, ___x56, ___x54_15);
                }
            }
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][0], ___x54_0);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][0], ___x54_1);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][0], ___x54_2);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][0], ___x54_3);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][0], ___x54_4);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][0], ___x54_5);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][0], ___x54_6);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][0], ___x54_7);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][0], ___x54_8);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][0], ___x54_9);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][0], ___x54_10);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][0], ___x54_11);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][0], ___x54_12);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][0], ___x54_13);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][0], ___x54_14);
            _mm512_store_ps(& ensemble22grad_inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][0], ___x54_15);
        }
    }
};
void backward8(float* _ensemble21grad_bias, float* _ensemble21grad) {
    float (* ensemble21grad)[256][16] = (float (*)[256][16]) _ensemble21grad;
    __assume_aligned(ensemble21grad, 64);
    float (* ensemble21grad_bias)[256][1][16] = (float (*)[256][1][16]) _ensemble21grad_bias;
    __assume_aligned(ensemble21grad_bias, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 256; _neuron_index_1_outer += 1) {
            for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                ensemble21grad_bias[_neuron_index_0][_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble21grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_1_inner];
            }
        }
    }
    
    #pragma omp parallel for simd
    for (int x = 0; x < 4096; ++x) {
      float sum = _ensemble21grad_bias[x];
      #pragma unroll
      for (int i = 1; i < 128; ++ i) {
        sum += _ensemble21grad_bias[i * 4096 + x];
      }
      _ensemble21grad_bias[x] = sum;
    }
};
void backward9(float* _ensemble20grad_weights, float* _ensemble20grad, float* _ensemble20inputs) {
    float (* ensemble20inputs)[16][9][9][16] = (float (*)[16][9][9][16]) _ensemble20inputs;
    __assume_aligned(ensemble20inputs, 64);
    float (* ensemble20grad)[256][16] = (float (*)[256][16]) _ensemble20grad;
    __assume_aligned(ensemble20grad, 64);
    float (* ensemble20grad_weights)[16][9][9][16][16] = (float (*)[16][9][9][16][16]) _ensemble20grad_weights;
    __assume_aligned(ensemble20grad_weights, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 256; _neuron_index_1_outer += 1) {
        for (int __unique_loopvar0_outer = 0; __unique_loopvar0_outer < 16; __unique_loopvar0_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 16) {
                for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                    __m512 ___x48_0 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 0)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_1 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 1)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_2 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 2)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_3 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 3)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_4 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 4)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_5 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 5)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_6 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 6)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_7 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 7)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_8 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 8)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_9 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 9)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_10 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 10)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_11 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 11)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_12 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 12)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_13 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 13)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_14 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 14)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    __m512 ___x48_15 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 15)][_neuron_index_1_outer][_neuron_index_1_inner]);
                    for (int __unique_loopvar1 = 0; __unique_loopvar1 < 9; __unique_loopvar1 += 1) {
                        for (int __unique_loopvar2 = 0; __unique_loopvar2 < 9; __unique_loopvar2 += 1) {
                            __m512 ___x49_0 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_1 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_2 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_3 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_4 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_5 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_6 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_7 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_8 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_9 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_10 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_11 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_12 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_13 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_14 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x49_15 = _mm512_load_ps(& ensemble20inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                            __m512 ___x50 = _mm512_load_ps(& ensemble20grad_weights[_neuron_index_1_outer][__unique_loopvar0_outer][__unique_loopvar1][__unique_loopvar2][_neuron_index_1_inner][0]);
                            ___x50 = _mm512_fmadd_ps(___x48_0, ___x49_0, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_1, ___x49_1, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_2, ___x49_2, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_3, ___x49_3, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_4, ___x49_4, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_5, ___x49_5, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_6, ___x49_6, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_7, ___x49_7, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_8, ___x49_8, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_9, ___x49_9, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_10, ___x49_10, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_11, ___x49_11, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_12, ___x49_12, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_13, ___x49_13, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_14, ___x49_14, ___x50);
                            ___x50 = _mm512_fmadd_ps(___x48_15, ___x49_15, ___x50);
                            _mm512_store_ps(& ensemble20grad_weights[_neuron_index_1_outer][__unique_loopvar0_outer][__unique_loopvar1][__unique_loopvar2][_neuron_index_1_inner][0], ___x50);
                        }
                    }
                }
            }
        }
    }
};
void backward10(float* _ensemble20grad, float* _ensemble20grad_inputs, float* _ensemble20weights) {
    float (* ensemble20weights)[16][9][9][16][16] = (float (*)[16][9][9][16][16]) _ensemble20weights;
    __assume_aligned(ensemble20weights, 64);
    float (* ensemble20grad_inputs)[16][9][9][16] = (float (*)[16][9][9][16]) _ensemble20grad_inputs;
    __assume_aligned(ensemble20grad_inputs, 64);
    float (* ensemble20grad)[256][16] = (float (*)[256][16]) _ensemble20grad;
    __assume_aligned(ensemble20grad, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 16) {
        for (int __unique_loopvar0_outer = 0; __unique_loopvar0_outer < 16; __unique_loopvar0_outer += 1) {
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 256; _neuron_index_1_outer += 1) {
                for (int __unique_loopvar1 = 0; __unique_loopvar1 < 9; __unique_loopvar1 += 1) {
                    for (int __unique_loopvar2 = 0; __unique_loopvar2 < 9; __unique_loopvar2 += 1) {
                        __m512 ___x47_0 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_1 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_2 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_3 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_4 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_5 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_6 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_7 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_8 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_9 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_10 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_11 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_12 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_13 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_14 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        __m512 ___x47_15 = _mm512_load_ps(& ensemble20grad_inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0]);
                        for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                            __m512 ___x45_0 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 0)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_1 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 1)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_2 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 2)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_3 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 3)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_4 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 4)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_5 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 5)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_6 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 6)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_7 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 7)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_8 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 8)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_9 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 9)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_10 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 10)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_11 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 11)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_12 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 12)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_13 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 13)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_14 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 14)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x45_15 = _mm512_set1_ps(ensemble20grad[(_neuron_index_0 + 15)][_neuron_index_1_outer][_neuron_index_1_inner]);
                            __m512 ___x46 = _mm512_load_ps(& ensemble20weights[_neuron_index_1_outer][__unique_loopvar0_outer][__unique_loopvar1][__unique_loopvar2][_neuron_index_1_inner][0]);
                            ___x47_0 = _mm512_fmadd_ps(___x45_0, ___x46, ___x47_0);
                            ___x47_1 = _mm512_fmadd_ps(___x45_1, ___x46, ___x47_1);
                            ___x47_2 = _mm512_fmadd_ps(___x45_2, ___x46, ___x47_2);
                            ___x47_3 = _mm512_fmadd_ps(___x45_3, ___x46, ___x47_3);
                            ___x47_4 = _mm512_fmadd_ps(___x45_4, ___x46, ___x47_4);
                            ___x47_5 = _mm512_fmadd_ps(___x45_5, ___x46, ___x47_5);
                            ___x47_6 = _mm512_fmadd_ps(___x45_6, ___x46, ___x47_6);
                            ___x47_7 = _mm512_fmadd_ps(___x45_7, ___x46, ___x47_7);
                            ___x47_8 = _mm512_fmadd_ps(___x45_8, ___x46, ___x47_8);
                            ___x47_9 = _mm512_fmadd_ps(___x45_9, ___x46, ___x47_9);
                            ___x47_10 = _mm512_fmadd_ps(___x45_10, ___x46, ___x47_10);
                            ___x47_11 = _mm512_fmadd_ps(___x45_11, ___x46, ___x47_11);
                            ___x47_12 = _mm512_fmadd_ps(___x45_12, ___x46, ___x47_12);
                            ___x47_13 = _mm512_fmadd_ps(___x45_13, ___x46, ___x47_13);
                            ___x47_14 = _mm512_fmadd_ps(___x45_14, ___x46, ___x47_14);
                            ___x47_15 = _mm512_fmadd_ps(___x45_15, ___x46, ___x47_15);
                        }
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 0)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_0);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 1)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_1);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 2)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_2);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 3)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_3);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 4)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_4);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 5)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_5);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 6)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_6);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 7)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_7);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 8)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_8);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 9)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_9);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 10)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_10);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 11)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_11);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 12)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_12);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 13)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_13);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 14)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_14);
                        _mm512_store_ps(& ensemble20grad_inputs[(_neuron_index_0 + 15)][__unique_loopvar0_outer][(__unique_loopvar1 * 1)][(__unique_loopvar2 * 1)][0], ___x47_15);
                    }
                }
            }
        }
    }
};
void backward11(float* _ensemble19grad_inputs, long* _ensemble19mask_k, float* _ensemble18grad, float* _ensemble18grad_inputs, float* _ensemble19grad, long* _ensemble19mask_j, float* _ensemble18inputs) {
    float (* ensemble18inputs)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble18inputs;
    __assume_aligned(ensemble18inputs, 64);
    long (* ensemble19mask_j)[16][9][9][16] = (long (*)[16][9][9][16]) _ensemble19mask_j;
    __assume_aligned(ensemble19mask_j, 64);
    float (* ensemble19grad)[16][9][9][16] = (float (*)[16][9][9][16]) _ensemble19grad;
    __assume_aligned(ensemble19grad, 64);
    float (* ensemble18grad_inputs)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble18grad_inputs;
    __assume_aligned(ensemble18grad_inputs, 64);
    float (* ensemble18grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble18grad;
    __assume_aligned(ensemble18grad, 64);
    long (* ensemble19mask_k)[16][9][9][16] = (long (*)[16][9][9][16]) _ensemble19mask_k;
    __assume_aligned(ensemble19mask_k, 64);
    float (* ensemble19grad_inputs)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble19grad_inputs;
    __assume_aligned(ensemble19grad_inputs, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 16; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 9; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 9; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        int _input_offset_1_outer = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) / 16;
                        int _input_offset_1_inner = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) % 16;
                        int in_y = _neuron_index_2 * 2 - 1;
                        int _input_offset_2 = in_y;
                        int in_x = _neuron_index_3 * 2 - 1;
                        int _input_offset_3 = in_x;
                        long j = ensemble19mask_j[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        long k = ensemble19mask_k[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        ensemble19grad_inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j + _input_offset_2, 0), 16)][MIN(MAX(k + _input_offset_3, 0), 16)][_input_offset_1_inner] += ensemble19grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                    }
                }
            }
            for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 17; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        if (ensemble18inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] > 0.0) {
                            ensemble18grad_inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble18grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        } else {
                            ensemble18grad_inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = 0.0;
                        };
                    }
                }
            }
        }
    }
};
void backward12(float* _ensemble17grad_bias, float* _ensemble16inputs, float* _ensemble16grad, float* _ensemble16grad_weights, float* _ensemble17grad) {
    float (* ensemble17grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble17grad;
    __assume_aligned(ensemble17grad, 64);
    float (* ensemble16grad_weights)[16][3][3][16][16] = (float (*)[16][3][3][16][16]) _ensemble16grad_weights;
    __assume_aligned(ensemble16grad_weights, 64);
    float (* ensemble16grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble16grad;
    __assume_aligned(ensemble16grad, 64);
    float (* ensemble16inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble16inputs;
    __assume_aligned(ensemble16inputs, 64);
    float (* ensemble17grad_bias)[1][16] = (float (*)[1][16]) _ensemble17grad_bias;
    __assume_aligned(ensemble17grad_bias, 64);
    #pragma omp parallel for
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 16; _neuron_index_1_outer += 1) {
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 17; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble17grad_bias[_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble17grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int i_outer = 0; i_outer < 16; i_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
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
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            __m512 ___x41_0 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x41_1 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x41_2 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x41_3 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x41_4 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x41_5 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x41_6 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x41_7 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x41_8 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x41_9 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x41_10 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x41_11 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x41_12 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x41_13 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x41_14 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x41_15 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x41_16 = _mm512_load_ps(& ensemble16inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x39 = _mm512_load_ps(& ensemble16grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x40_0 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 0)][_neuron_index_1_inner]);
                                __m512 ___x40_1 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x40_2 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 2)][_neuron_index_1_inner]);
                                __m512 ___x40_3 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 3)][_neuron_index_1_inner]);
                                __m512 ___x40_4 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 4)][_neuron_index_1_inner]);
                                __m512 ___x40_5 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 5)][_neuron_index_1_inner]);
                                __m512 ___x40_6 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 6)][_neuron_index_1_inner]);
                                __m512 ___x40_7 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 7)][_neuron_index_1_inner]);
                                __m512 ___x40_8 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 8)][_neuron_index_1_inner]);
                                __m512 ___x40_9 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 9)][_neuron_index_1_inner]);
                                __m512 ___x40_10 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 10)][_neuron_index_1_inner]);
                                __m512 ___x40_11 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 11)][_neuron_index_1_inner]);
                                __m512 ___x40_12 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 12)][_neuron_index_1_inner]);
                                __m512 ___x40_13 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 13)][_neuron_index_1_inner]);
                                __m512 ___x40_14 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 14)][_neuron_index_1_inner]);
                                __m512 ___x40_15 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 15)][_neuron_index_1_inner]);
                                __m512 ___x40_16 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 16)][_neuron_index_1_inner]);
                                ___x39 = _mm512_fmadd_ps(___x40_0, ___x41_0, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_1, ___x41_1, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_2, ___x41_2, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_3, ___x41_3, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_4, ___x41_4, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_5, ___x41_5, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_6, ___x41_6, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_7, ___x41_7, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_8, ___x41_8, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_9, ___x41_9, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_10, ___x41_10, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_11, ___x41_11, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_12, ___x41_12, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_13, ___x41_13, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_14, ___x41_14, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_15, ___x41_15, ___x39);
                                ___x39 = _mm512_fmadd_ps(___x40_16, ___x41_16, ___x39);
                                _mm512_store_ps(& ensemble16grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0], ___x39);
                            }
                        }
                    }
                }
            }
        }
    }
};
void backward13(float* _ensemble15grad, float* _ensemble16weights, float* _ensemble15inputs, float* _ensemble15grad_inputs, float* _ensemble16grad, float* _ensemble16grad_inputs) {
    float (* ensemble16grad_inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble16grad_inputs;
    __assume_aligned(ensemble16grad_inputs, 64);
    float (* ensemble16grad)[16][17][17][16] = (float (*)[16][17][17][16]) _ensemble16grad;
    __assume_aligned(ensemble16grad, 64);
    float (* ensemble15grad_inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble15grad_inputs;
    __assume_aligned(ensemble15grad_inputs, 64);
    float (* ensemble15inputs)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble15inputs;
    __assume_aligned(ensemble15inputs, 64);
    float (* ensemble16weights)[16][3][3][16][16] = (float (*)[16][3][3][16][16]) _ensemble16weights;
    __assume_aligned(ensemble16weights, 64);
    float (* ensemble15grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble15grad;
    __assume_aligned(ensemble15grad, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int i_outer = 0; i_outer < 16; i_outer += 1) {
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 16; _neuron_index_1_outer += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
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
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            __m512 ___x36_0 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x36_1 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x36_2 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x36_3 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x36_4 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x36_5 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x36_6 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x36_7 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x36_8 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x36_9 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x36_10 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x36_11 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x36_12 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x36_13 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x36_14 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x36_15 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x36_16 = _mm512_load_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x37 = _mm512_load_ps(& ensemble16weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x38_0 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 0)][_neuron_index_1_inner]);
                                __m512 ___x38_1 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x38_2 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 2)][_neuron_index_1_inner]);
                                __m512 ___x38_3 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 3)][_neuron_index_1_inner]);
                                __m512 ___x38_4 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 4)][_neuron_index_1_inner]);
                                __m512 ___x38_5 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 5)][_neuron_index_1_inner]);
                                __m512 ___x38_6 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 6)][_neuron_index_1_inner]);
                                __m512 ___x38_7 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 7)][_neuron_index_1_inner]);
                                __m512 ___x38_8 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 8)][_neuron_index_1_inner]);
                                __m512 ___x38_9 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 9)][_neuron_index_1_inner]);
                                __m512 ___x38_10 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 10)][_neuron_index_1_inner]);
                                __m512 ___x38_11 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 11)][_neuron_index_1_inner]);
                                __m512 ___x38_12 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 12)][_neuron_index_1_inner]);
                                __m512 ___x38_13 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 13)][_neuron_index_1_inner]);
                                __m512 ___x38_14 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 14)][_neuron_index_1_inner]);
                                __m512 ___x38_15 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 15)][_neuron_index_1_inner]);
                                __m512 ___x38_16 = _mm512_set1_ps(ensemble16grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 16)][_neuron_index_1_inner]);
                                ___x36_0 = _mm512_fmadd_ps(___x38_0, ___x37, ___x36_0);
                                ___x36_1 = _mm512_fmadd_ps(___x38_1, ___x37, ___x36_1);
                                ___x36_2 = _mm512_fmadd_ps(___x38_2, ___x37, ___x36_2);
                                ___x36_3 = _mm512_fmadd_ps(___x38_3, ___x37, ___x36_3);
                                ___x36_4 = _mm512_fmadd_ps(___x38_4, ___x37, ___x36_4);
                                ___x36_5 = _mm512_fmadd_ps(___x38_5, ___x37, ___x36_5);
                                ___x36_6 = _mm512_fmadd_ps(___x38_6, ___x37, ___x36_6);
                                ___x36_7 = _mm512_fmadd_ps(___x38_7, ___x37, ___x36_7);
                                ___x36_8 = _mm512_fmadd_ps(___x38_8, ___x37, ___x36_8);
                                ___x36_9 = _mm512_fmadd_ps(___x38_9, ___x37, ___x36_9);
                                ___x36_10 = _mm512_fmadd_ps(___x38_10, ___x37, ___x36_10);
                                ___x36_11 = _mm512_fmadd_ps(___x38_11, ___x37, ___x36_11);
                                ___x36_12 = _mm512_fmadd_ps(___x38_12, ___x37, ___x36_12);
                                ___x36_13 = _mm512_fmadd_ps(___x38_13, ___x37, ___x36_13);
                                ___x36_14 = _mm512_fmadd_ps(___x38_14, ___x37, ___x36_14);
                                ___x36_15 = _mm512_fmadd_ps(___x38_15, ___x37, ___x36_15);
                                ___x36_16 = _mm512_fmadd_ps(___x38_16, ___x37, ___x36_16);
                            }
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0], ___x36_0);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0], ___x36_1);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0], ___x36_2);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0], ___x36_3);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0], ___x36_4);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0], ___x36_5);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0], ___x36_6);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0], ___x36_7);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0], ___x36_8);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0], ___x36_9);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0], ___x36_10);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0], ___x36_11);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0], ___x36_12);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0], ___x36_13);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0], ___x36_14);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0], ___x36_15);
                            _mm512_store_ps(& ensemble16grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0], ___x36_16);
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 16; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 17; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        if (ensemble15inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] > 0.0) {
                            ensemble15grad_inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = ensemble15grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner];
                        } else {
                            ensemble15grad_inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = 0.0;
                        };
                    }
                }
            }
        }
    }
};
void backward14(float* _ensemble14grad_bias, float* _ensemble13grad_weights, float* _ensemble13inputs, float* _ensemble14grad, float* _ensemble13grad) {
    float (* ensemble13grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble13grad;
    __assume_aligned(ensemble13grad, 64);
    float (* ensemble14grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble14grad;
    __assume_aligned(ensemble14grad, 64);
    float (* ensemble13inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble13inputs;
    __assume_aligned(ensemble13inputs, 64);
    float (* ensemble13grad_weights)[24][3][3][16][16] = (float (*)[24][3][3][16][16]) _ensemble13grad_weights;
    __assume_aligned(ensemble13grad_weights, 64);
    float (* ensemble14grad_bias)[1][16] = (float (*)[1][16]) _ensemble14grad_bias;
    __assume_aligned(ensemble14grad_bias, 64);
    #pragma omp parallel for
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 16; _neuron_index_1_outer += 1) {
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 17; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble14grad_bias[_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble14grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int i_outer = 0; i_outer < 24; i_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
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
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            __m512 ___x30_0 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x30_1 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x30_2 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x30_3 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x30_4 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x30_5 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x30_6 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x30_7 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x30_8 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x30_9 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x30_10 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x30_11 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x30_12 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x30_13 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x30_14 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x30_15 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x30_16 = _mm512_load_ps(& ensemble13inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x31 = _mm512_load_ps(& ensemble13grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x32_0 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_1 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 1 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_2 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 2 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_3 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 3 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_4 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 4 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_5 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 5 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_6 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 6 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_7 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 7 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_8 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 8 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_9 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 9 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_10 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 10 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_11 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 11 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_12 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 12 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_13 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 13 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_14 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 14 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_15 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 15 + 1)][_neuron_index_1_inner]);
                                __m512 ___x32_16 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 16 + 1)][_neuron_index_1_inner]);
                                ___x31 = _mm512_fmadd_ps(___x32_0, ___x30_0, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_1, ___x30_1, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_2, ___x30_2, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_3, ___x30_3, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_4, ___x30_4, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_5, ___x30_5, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_6, ___x30_6, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_7, ___x30_7, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_8, ___x30_8, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_9, ___x30_9, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_10, ___x30_10, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_11, ___x30_11, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_12, ___x30_12, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_13, ___x30_13, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_14, ___x30_14, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_15, ___x30_15, ___x31);
                                ___x31 = _mm512_fmadd_ps(___x32_16, ___x30_16, ___x31);
                                _mm512_store_ps(& ensemble13grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0], ___x31);
                            }
                        }
                    }
                }
            }
        }
    }
};
void backward15(float* _ensemble12grad_inputs, float* _ensemble12inputs, float* _ensemble13grad_inputs, float* _ensemble13grad, float* _ensemble12grad, float* _ensemble13weights) {
    float (* ensemble13weights)[24][3][3][16][16] = (float (*)[24][3][3][16][16]) _ensemble13weights;
    __assume_aligned(ensemble13weights, 64);
    float (* ensemble12grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble12grad;
    __assume_aligned(ensemble12grad, 64);
    float (* ensemble13grad)[16][19][19][16] = (float (*)[16][19][19][16]) _ensemble13grad;
    __assume_aligned(ensemble13grad, 64);
    float (* ensemble13grad_inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble13grad_inputs;
    __assume_aligned(ensemble13grad_inputs, 64);
    float (* ensemble12inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble12inputs;
    __assume_aligned(ensemble12inputs, 64);
    float (* ensemble12grad_inputs)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble12grad_inputs;
    __assume_aligned(ensemble12grad_inputs, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int i_outer = 0; i_outer < 24; i_outer += 1) {
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 16; _neuron_index_1_outer += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
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
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            __m512 ___x27_0 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x27_1 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x27_2 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x27_3 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x27_4 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x27_5 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x27_6 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x27_7 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x27_8 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x27_9 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x27_10 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x27_11 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x27_12 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x27_13 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x27_14 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x27_15 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x27_16 = _mm512_load_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x28 = _mm512_load_ps(& ensemble13weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x29_0 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_1 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 1 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_2 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 2 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_3 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 3 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_4 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 4 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_5 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 5 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_6 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 6 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_7 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 7 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_8 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 8 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_9 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 9 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_10 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 10 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_11 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 11 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_12 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 12 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_13 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 13 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_14 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 14 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_15 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 15 + 1)][_neuron_index_1_inner]);
                                __m512 ___x29_16 = _mm512_set1_ps(ensemble13grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 16 + 1)][_neuron_index_1_inner]);
                                ___x27_0 = _mm512_fmadd_ps(___x29_0, ___x28, ___x27_0);
                                ___x27_1 = _mm512_fmadd_ps(___x29_1, ___x28, ___x27_1);
                                ___x27_2 = _mm512_fmadd_ps(___x29_2, ___x28, ___x27_2);
                                ___x27_3 = _mm512_fmadd_ps(___x29_3, ___x28, ___x27_3);
                                ___x27_4 = _mm512_fmadd_ps(___x29_4, ___x28, ___x27_4);
                                ___x27_5 = _mm512_fmadd_ps(___x29_5, ___x28, ___x27_5);
                                ___x27_6 = _mm512_fmadd_ps(___x29_6, ___x28, ___x27_6);
                                ___x27_7 = _mm512_fmadd_ps(___x29_7, ___x28, ___x27_7);
                                ___x27_8 = _mm512_fmadd_ps(___x29_8, ___x28, ___x27_8);
                                ___x27_9 = _mm512_fmadd_ps(___x29_9, ___x28, ___x27_9);
                                ___x27_10 = _mm512_fmadd_ps(___x29_10, ___x28, ___x27_10);
                                ___x27_11 = _mm512_fmadd_ps(___x29_11, ___x28, ___x27_11);
                                ___x27_12 = _mm512_fmadd_ps(___x29_12, ___x28, ___x27_12);
                                ___x27_13 = _mm512_fmadd_ps(___x29_13, ___x28, ___x27_13);
                                ___x27_14 = _mm512_fmadd_ps(___x29_14, ___x28, ___x27_14);
                                ___x27_15 = _mm512_fmadd_ps(___x29_15, ___x28, ___x27_15);
                                ___x27_16 = _mm512_fmadd_ps(___x29_16, ___x28, ___x27_16);
                            }
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0], ___x27_0);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0], ___x27_1);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0], ___x27_2);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0], ___x27_3);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0], ___x27_4);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0], ___x27_5);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0], ___x27_6);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0], ___x27_7);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0], ___x27_8);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0], ___x27_9);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0], ___x27_10);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0], ___x27_11);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0], ___x27_12);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0], ___x27_13);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0], ___x27_14);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0], ___x27_15);
                            _mm512_store_ps(& ensemble13grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0], ___x27_16);
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 24; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 17; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        if (ensemble12inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] > 0.0) {
                            ensemble12grad_inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = ensemble12grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner];
                        } else {
                            ensemble12grad_inputs[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner] = 0.0;
                        };
                    }
                }
            }
        }
    }
};
void backward16(float* _ensemble10inputs, float* _ensemble11grad, float* _ensemble11grad_bias, float* _ensemble10grad_weights, float* _ensemble10grad) {
    float (* ensemble10grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble10grad;
    __assume_aligned(ensemble10grad, 64);
    float (* ensemble10grad_weights)[12][3][3][16][16] = (float (*)[12][3][3][16][16]) _ensemble10grad_weights;
    __assume_aligned(ensemble10grad_weights, 64);
    float (* ensemble11grad_bias)[1][16] = (float (*)[1][16]) _ensemble11grad_bias;
    __assume_aligned(ensemble11grad_bias, 64);
    float (* ensemble11grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble11grad;
    __assume_aligned(ensemble11grad, 64);
    float (* ensemble10inputs)[12][19][19][16] = (float (*)[12][19][19][16]) _ensemble10inputs;
    __assume_aligned(ensemble10inputs, 64);
    #pragma omp parallel for
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 24; _neuron_index_1_outer += 1) {
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 17; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble11grad_bias[_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble11grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(_neuron_index_3 + 1)][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int i_outer = 0; i_outer < 12; i_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
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
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            __m512 ___x22_0 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x22_1 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x22_2 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x22_3 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x22_4 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x22_5 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x22_6 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x22_7 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x22_8 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x22_9 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x22_10 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x22_11 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x22_12 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x22_13 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x22_14 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x22_15 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x22_16 = _mm512_load_ps(& ensemble10inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x21_0 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_1 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 1 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_2 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 2 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_3 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 3 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_4 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 4 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_5 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 5 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_6 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 6 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_7 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 7 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_8 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 8 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_9 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 9 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_10 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 10 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_11 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 11 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_12 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 12 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_13 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 13 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_14 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 14 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_15 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 15 + 1)][_neuron_index_1_inner]);
                                __m512 ___x21_16 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 16 + 1)][_neuron_index_1_inner]);
                                __m512 ___x23 = _mm512_load_ps(& ensemble10grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                ___x23 = _mm512_fmadd_ps(___x21_0, ___x22_0, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_1, ___x22_1, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_2, ___x22_2, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_3, ___x22_3, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_4, ___x22_4, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_5, ___x22_5, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_6, ___x22_6, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_7, ___x22_7, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_8, ___x22_8, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_9, ___x22_9, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_10, ___x22_10, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_11, ___x22_11, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_12, ___x22_12, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_13, ___x22_13, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_14, ___x22_14, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_15, ___x22_15, ___x23);
                                ___x23 = _mm512_fmadd_ps(___x21_16, ___x22_16, ___x23);
                                _mm512_store_ps(& ensemble10grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0], ___x23);
                            }
                        }
                    }
                }
            }
        }
    }
};
void backward17(float* _ensemble10grad, float* _ensemble10grad_inputs, float* _ensemble10weights, float* _ensemble9grad_inputs, float* _ensemble9grad, long* _ensemble9mask_j, float* _ensemble8grad_inputs, float* _ensemble8inputs, float* _ensemble8grad, long* _ensemble9mask_k) {
    long (* ensemble9mask_k)[12][15][15][16] = (long (*)[12][15][15][16]) _ensemble9mask_k;
    __assume_aligned(ensemble9mask_k, 64);
    float (* ensemble8grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble8grad;
    __assume_aligned(ensemble8grad, 64);
    float (* ensemble8inputs)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble8inputs;
    __assume_aligned(ensemble8inputs, 64);
    float (* ensemble8grad_inputs)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble8grad_inputs;
    __assume_aligned(ensemble8grad_inputs, 64);
    long (* ensemble9mask_j)[12][15][15][16] = (long (*)[12][15][15][16]) _ensemble9mask_j;
    __assume_aligned(ensemble9mask_j, 64);
    float (* ensemble9grad)[12][19][19][16] = (float (*)[12][19][19][16]) _ensemble9grad;
    __assume_aligned(ensemble9grad, 64);
    float (* ensemble9grad_inputs)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble9grad_inputs;
    __assume_aligned(ensemble9grad_inputs, 64);
    float (* ensemble10weights)[12][3][3][16][16] = (float (*)[12][3][3][16][16]) _ensemble10weights;
    __assume_aligned(ensemble10weights, 64);
    float (* ensemble10grad_inputs)[12][19][19][16] = (float (*)[12][19][19][16]) _ensemble10grad_inputs;
    __assume_aligned(ensemble10grad_inputs, 64);
    float (* ensemble10grad)[24][19][19][16] = (float (*)[24][19][19][16]) _ensemble10grad;
    __assume_aligned(ensemble10grad, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int i_outer = 0; i_outer < 12; i_outer += 1) {
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 24; _neuron_index_1_outer += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 17; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
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
                    for (int j = 0; j < 3; j += 1) {
                        for (int k = 0; k < 3; k += 1) {
                            __m512 ___x20_0 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x20_1 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x20_2 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x20_3 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x20_4 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x20_5 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x20_6 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x20_7 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x20_8 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x20_9 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x20_10 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x20_11 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x20_12 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x20_13 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x20_14 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x20_15 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x20_16 = _mm512_load_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x18 = _mm512_load_ps(& ensemble10weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x19_0 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_1 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 1 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_2 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 2 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_3 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 3 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_4 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 4 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_5 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 5 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_6 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 6 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_7 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 7 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_8 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 8 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_9 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 9 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_10 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 10 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_11 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 11 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_12 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 12 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_13 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 13 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_14 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 14 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_15 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 15 + 1)][_neuron_index_1_inner]);
                                __m512 ___x19_16 = _mm512_set1_ps(ensemble10grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 1)][(0 + 16 + 1)][_neuron_index_1_inner]);
                                ___x20_0 = _mm512_fmadd_ps(___x19_0, ___x18, ___x20_0);
                                ___x20_1 = _mm512_fmadd_ps(___x19_1, ___x18, ___x20_1);
                                ___x20_2 = _mm512_fmadd_ps(___x19_2, ___x18, ___x20_2);
                                ___x20_3 = _mm512_fmadd_ps(___x19_3, ___x18, ___x20_3);
                                ___x20_4 = _mm512_fmadd_ps(___x19_4, ___x18, ___x20_4);
                                ___x20_5 = _mm512_fmadd_ps(___x19_5, ___x18, ___x20_5);
                                ___x20_6 = _mm512_fmadd_ps(___x19_6, ___x18, ___x20_6);
                                ___x20_7 = _mm512_fmadd_ps(___x19_7, ___x18, ___x20_7);
                                ___x20_8 = _mm512_fmadd_ps(___x19_8, ___x18, ___x20_8);
                                ___x20_9 = _mm512_fmadd_ps(___x19_9, ___x18, ___x20_9);
                                ___x20_10 = _mm512_fmadd_ps(___x19_10, ___x18, ___x20_10);
                                ___x20_11 = _mm512_fmadd_ps(___x19_11, ___x18, ___x20_11);
                                ___x20_12 = _mm512_fmadd_ps(___x19_12, ___x18, ___x20_12);
                                ___x20_13 = _mm512_fmadd_ps(___x19_13, ___x18, ___x20_13);
                                ___x20_14 = _mm512_fmadd_ps(___x19_14, ___x18, ___x20_14);
                                ___x20_15 = _mm512_fmadd_ps(___x19_15, ___x18, ___x20_15);
                                ___x20_16 = _mm512_fmadd_ps(___x19_16, ___x18, ___x20_16);
                            }
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0], ___x20_0);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0], ___x20_1);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0], ___x20_2);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0], ___x20_3);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0], ___x20_4);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0], ___x20_5);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0], ___x20_6);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0], ___x20_7);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0], ___x20_8);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0], ___x20_9);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0], ___x20_10);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0], ___x20_11);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0], ___x20_12);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0], ___x20_13);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0], ___x20_14);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0], ___x20_15);
                            _mm512_store_ps(& ensemble10grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0], ___x20_16);
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 12; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 15; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 15; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        int _input_offset_1_outer = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) / 16;
                        int _input_offset_1_inner = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) % 16;
                        int in_y = _neuron_index_2 * 2 - 1;
                        int _input_offset_2 = in_y;
                        int in_x = _neuron_index_3 * 2 - 1;
                        int _input_offset_3 = in_x;
                        long j = ensemble9mask_j[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        long k = ensemble9mask_k[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        ensemble9grad_inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j + _input_offset_2, 0), 27)][MIN(MAX(k + _input_offset_3, 0), 27)][_input_offset_1_inner] += ensemble9grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 2)][(_neuron_index_3 + 2)][_neuron_index_1_inner];
                    }
                }
            }
            for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        if (ensemble8inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] > 0.0) {
                            ensemble8grad_inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = ensemble8grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        } else {
                            ensemble8grad_inputs[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner] = 0.0;
                        };
                    }
                }
            }
        }
    }
};
void backward18(float* _ensemble6grad, float* _ensemble6inputs, float* _ensemble6grad_weights, float* _ensemble7grad_bias, float* _ensemble7grad) {
    float (* ensemble7grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble7grad;
    __assume_aligned(ensemble7grad, 64);
    float (* ensemble7grad_bias)[1][16] = (float (*)[1][16]) _ensemble7grad_bias;
    __assume_aligned(ensemble7grad_bias, 64);
    float (* ensemble6grad_weights)[4][5][5][16][16] = (float (*)[4][5][5][16][16]) _ensemble6grad_weights;
    __assume_aligned(ensemble6grad_weights, 64);
    float (* ensemble6inputs)[4][32][32][16] = (float (*)[4][32][32][16]) _ensemble6inputs;
    __assume_aligned(ensemble6inputs, 64);
    float (* ensemble6grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble6grad;
    __assume_aligned(ensemble6grad, 64);
    #pragma omp parallel for
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 12; _neuron_index_1_outer += 1) {
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble7grad_bias[_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble7grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int i_outer = 0; i_outer < 4; i_outer += 1) {
            for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
                    int in_x_17 = (0 + 17) * 1;
                    int in_x_18 = (0 + 18) * 1;
                    int in_x_19 = (0 + 19) * 1;
                    int in_x_20 = (0 + 20) * 1;
                    int in_x_21 = (0 + 21) * 1;
                    int in_x_22 = (0 + 22) * 1;
                    int in_x_23 = (0 + 23) * 1;
                    int in_x_24 = (0 + 24) * 1;
                    int in_x_25 = (0 + 25) * 1;
                    int in_x_26 = (0 + 26) * 1;
                    int in_x_27 = (0 + 27) * 1;
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
                    for (int j = 0; j < 5; j += 1) {
                        for (int k = 0; k < 5; k += 1) {
                            __m512 ___x13_0 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x13_1 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x13_2 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x13_3 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x13_4 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x13_5 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x13_6 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x13_7 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x13_8 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x13_9 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x13_10 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x13_11 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x13_12 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x13_13 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x13_14 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x13_15 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x13_16 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            __m512 ___x13_17 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 17) * 1)][0]);
                            __m512 ___x13_18 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 18) * 1)][0]);
                            __m512 ___x13_19 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 19) * 1)][0]);
                            __m512 ___x13_20 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 20) * 1)][0]);
                            __m512 ___x13_21 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 21) * 1)][0]);
                            __m512 ___x13_22 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 22) * 1)][0]);
                            __m512 ___x13_23 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 23) * 1)][0]);
                            __m512 ___x13_24 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 24) * 1)][0]);
                            __m512 ___x13_25 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 25) * 1)][0]);
                            __m512 ___x13_26 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 26) * 1)][0]);
                            __m512 ___x13_27 = _mm512_load_ps(& ensemble6inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 27) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x12_0 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 0)][_neuron_index_1_inner]);
                                __m512 ___x12_1 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x12_2 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 2)][_neuron_index_1_inner]);
                                __m512 ___x12_3 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 3)][_neuron_index_1_inner]);
                                __m512 ___x12_4 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 4)][_neuron_index_1_inner]);
                                __m512 ___x12_5 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 5)][_neuron_index_1_inner]);
                                __m512 ___x12_6 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 6)][_neuron_index_1_inner]);
                                __m512 ___x12_7 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 7)][_neuron_index_1_inner]);
                                __m512 ___x12_8 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 8)][_neuron_index_1_inner]);
                                __m512 ___x12_9 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 9)][_neuron_index_1_inner]);
                                __m512 ___x12_10 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 10)][_neuron_index_1_inner]);
                                __m512 ___x12_11 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 11)][_neuron_index_1_inner]);
                                __m512 ___x12_12 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 12)][_neuron_index_1_inner]);
                                __m512 ___x12_13 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 13)][_neuron_index_1_inner]);
                                __m512 ___x12_14 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 14)][_neuron_index_1_inner]);
                                __m512 ___x12_15 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 15)][_neuron_index_1_inner]);
                                __m512 ___x12_16 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 16)][_neuron_index_1_inner]);
                                __m512 ___x12_17 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 17)][_neuron_index_1_inner]);
                                __m512 ___x12_18 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 18)][_neuron_index_1_inner]);
                                __m512 ___x12_19 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 19)][_neuron_index_1_inner]);
                                __m512 ___x12_20 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 20)][_neuron_index_1_inner]);
                                __m512 ___x12_21 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 21)][_neuron_index_1_inner]);
                                __m512 ___x12_22 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 22)][_neuron_index_1_inner]);
                                __m512 ___x12_23 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 23)][_neuron_index_1_inner]);
                                __m512 ___x12_24 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 24)][_neuron_index_1_inner]);
                                __m512 ___x12_25 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 25)][_neuron_index_1_inner]);
                                __m512 ___x12_26 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 26)][_neuron_index_1_inner]);
                                __m512 ___x12_27 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 27)][_neuron_index_1_inner]);
                                __m512 ___x14 = _mm512_load_ps(& ensemble6grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                ___x14 = _mm512_fmadd_ps(___x12_0, ___x13_0, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_1, ___x13_1, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_2, ___x13_2, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_3, ___x13_3, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_4, ___x13_4, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_5, ___x13_5, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_6, ___x13_6, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_7, ___x13_7, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_8, ___x13_8, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_9, ___x13_9, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_10, ___x13_10, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_11, ___x13_11, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_12, ___x13_12, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_13, ___x13_13, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_14, ___x13_14, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_15, ___x13_15, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_16, ___x13_16, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_17, ___x13_17, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_18, ___x13_18, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_19, ___x13_19, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_20, ___x13_20, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_21, ___x13_21, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_22, ___x13_22, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_23, ___x13_23, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_24, ___x13_24, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_25, ___x13_25, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_26, ___x13_26, ___x14);
                                ___x14 = _mm512_fmadd_ps(___x12_27, ___x13_27, ___x14);
                                _mm512_store_ps(& ensemble6grad_weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0], ___x14);
                            }
                        }
                    }
                }
            }
        }
    }
};
void backward19(long* _ensemble5mask_k, float* _ensemble5grad, long* _ensemble5mask_j, float* _ensemble4inputs, float* _ensemble6weights, float* _ensemble4grad_inputs, float* _ensemble6grad, float* _ensemble4grad, float* _ensemble5grad_inputs, float* _ensemble6grad_inputs) {
    float (* ensemble6grad_inputs)[4][32][32][16] = (float (*)[4][32][32][16]) _ensemble6grad_inputs;
    __assume_aligned(ensemble6grad_inputs, 64);
    float (* ensemble5grad_inputs)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble5grad_inputs;
    __assume_aligned(ensemble5grad_inputs, 64);
    float (* ensemble4grad)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble4grad;
    __assume_aligned(ensemble4grad, 64);
    float (* ensemble6grad)[12][28][28][16] = (float (*)[12][28][28][16]) _ensemble6grad;
    __assume_aligned(ensemble6grad, 64);
    float (* ensemble4grad_inputs)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble4grad_inputs;
    __assume_aligned(ensemble4grad_inputs, 64);
    float (* ensemble6weights)[4][5][5][16][16] = (float (*)[4][5][5][16][16]) _ensemble6weights;
    __assume_aligned(ensemble6weights, 64);
    float (* ensemble4inputs)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble4inputs;
    __assume_aligned(ensemble4inputs, 64);
    long (* ensemble5mask_j)[4][28][28][16] = (long (*)[4][28][28][16]) _ensemble5mask_j;
    __assume_aligned(ensemble5mask_j, 64);
    float (* ensemble5grad)[4][32][32][16] = (float (*)[4][32][32][16]) _ensemble5grad;
    __assume_aligned(ensemble5grad, 64);
    long (* ensemble5mask_k)[4][28][28][16] = (long (*)[4][28][28][16]) _ensemble5mask_k;
    __assume_aligned(ensemble5mask_k, 64);
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int i_outer = 0; i_outer < 4; i_outer += 1) {
            for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 12; _neuron_index_1_outer += 1) {
                for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
                    int in_y = _neuron_index_2 * 1;
                    int _input_offset_2 = in_y;
                    int in_x_0 = (0 + 0) * 1;
                    int in_x_1 = (0 + 1) * 1;
                    int in_x_2 = (0 + 2) * 1;
                    int in_x_3 = (0 + 3) * 1;
                    int in_x_4 = (0 + 4) * 1;
                    int in_x_5 = (0 + 5) * 1;
                    int in_x_6 = (0 + 6) * 1;
                    int in_x_7 = (0 + 7) * 1;
                    int in_x_8 = (0 + 8) * 1;
                    int in_x_9 = (0 + 9) * 1;
                    int in_x_10 = (0 + 10) * 1;
                    int in_x_11 = (0 + 11) * 1;
                    int in_x_12 = (0 + 12) * 1;
                    int in_x_13 = (0 + 13) * 1;
                    int in_x_14 = (0 + 14) * 1;
                    int in_x_15 = (0 + 15) * 1;
                    int in_x_16 = (0 + 16) * 1;
                    int in_x_17 = (0 + 17) * 1;
                    int in_x_18 = (0 + 18) * 1;
                    int in_x_19 = (0 + 19) * 1;
                    int in_x_20 = (0 + 20) * 1;
                    int in_x_21 = (0 + 21) * 1;
                    int in_x_22 = (0 + 22) * 1;
                    int in_x_23 = (0 + 23) * 1;
                    int in_x_24 = (0 + 24) * 1;
                    int in_x_25 = (0 + 25) * 1;
                    int in_x_26 = (0 + 26) * 1;
                    int in_x_27 = (0 + 27) * 1;
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
                    for (int j = 0; j < 5; j += 1) {
                        for (int k = 0; k < 5; k += 1) {
                            __m512 ___x9_0 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0]);
                            __m512 ___x9_1 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0]);
                            __m512 ___x9_2 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0]);
                            __m512 ___x9_3 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0]);
                            __m512 ___x9_4 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0]);
                            __m512 ___x9_5 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0]);
                            __m512 ___x9_6 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0]);
                            __m512 ___x9_7 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0]);
                            __m512 ___x9_8 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0]);
                            __m512 ___x9_9 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0]);
                            __m512 ___x9_10 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0]);
                            __m512 ___x9_11 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0]);
                            __m512 ___x9_12 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0]);
                            __m512 ___x9_13 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0]);
                            __m512 ___x9_14 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0]);
                            __m512 ___x9_15 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0]);
                            __m512 ___x9_16 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0]);
                            __m512 ___x9_17 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 17) * 1)][0]);
                            __m512 ___x9_18 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 18) * 1)][0]);
                            __m512 ___x9_19 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 19) * 1)][0]);
                            __m512 ___x9_20 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 20) * 1)][0]);
                            __m512 ___x9_21 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 21) * 1)][0]);
                            __m512 ___x9_22 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 22) * 1)][0]);
                            __m512 ___x9_23 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 23) * 1)][0]);
                            __m512 ___x9_24 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 24) * 1)][0]);
                            __m512 ___x9_25 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 25) * 1)][0]);
                            __m512 ___x9_26 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 26) * 1)][0]);
                            __m512 ___x9_27 = _mm512_load_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 27) * 1)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x10 = _mm512_load_ps(& ensemble6weights[_neuron_index_1_outer][i_outer][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x11_0 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 0)][_neuron_index_1_inner]);
                                __m512 ___x11_1 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 1)][_neuron_index_1_inner]);
                                __m512 ___x11_2 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 2)][_neuron_index_1_inner]);
                                __m512 ___x11_3 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 3)][_neuron_index_1_inner]);
                                __m512 ___x11_4 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 4)][_neuron_index_1_inner]);
                                __m512 ___x11_5 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 5)][_neuron_index_1_inner]);
                                __m512 ___x11_6 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 6)][_neuron_index_1_inner]);
                                __m512 ___x11_7 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 7)][_neuron_index_1_inner]);
                                __m512 ___x11_8 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 8)][_neuron_index_1_inner]);
                                __m512 ___x11_9 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 9)][_neuron_index_1_inner]);
                                __m512 ___x11_10 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 10)][_neuron_index_1_inner]);
                                __m512 ___x11_11 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 11)][_neuron_index_1_inner]);
                                __m512 ___x11_12 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 12)][_neuron_index_1_inner]);
                                __m512 ___x11_13 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 13)][_neuron_index_1_inner]);
                                __m512 ___x11_14 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 14)][_neuron_index_1_inner]);
                                __m512 ___x11_15 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 15)][_neuron_index_1_inner]);
                                __m512 ___x11_16 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 16)][_neuron_index_1_inner]);
                                __m512 ___x11_17 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 17)][_neuron_index_1_inner]);
                                __m512 ___x11_18 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 18)][_neuron_index_1_inner]);
                                __m512 ___x11_19 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 19)][_neuron_index_1_inner]);
                                __m512 ___x11_20 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 20)][_neuron_index_1_inner]);
                                __m512 ___x11_21 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 21)][_neuron_index_1_inner]);
                                __m512 ___x11_22 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 22)][_neuron_index_1_inner]);
                                __m512 ___x11_23 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 23)][_neuron_index_1_inner]);
                                __m512 ___x11_24 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 24)][_neuron_index_1_inner]);
                                __m512 ___x11_25 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 25)][_neuron_index_1_inner]);
                                __m512 ___x11_26 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 26)][_neuron_index_1_inner]);
                                __m512 ___x11_27 = _mm512_set1_ps(ensemble6grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(0 + 27)][_neuron_index_1_inner]);
                                ___x9_0 = _mm512_fmadd_ps(___x11_0, ___x10, ___x9_0);
                                ___x9_1 = _mm512_fmadd_ps(___x11_1, ___x10, ___x9_1);
                                ___x9_2 = _mm512_fmadd_ps(___x11_2, ___x10, ___x9_2);
                                ___x9_3 = _mm512_fmadd_ps(___x11_3, ___x10, ___x9_3);
                                ___x9_4 = _mm512_fmadd_ps(___x11_4, ___x10, ___x9_4);
                                ___x9_5 = _mm512_fmadd_ps(___x11_5, ___x10, ___x9_5);
                                ___x9_6 = _mm512_fmadd_ps(___x11_6, ___x10, ___x9_6);
                                ___x9_7 = _mm512_fmadd_ps(___x11_7, ___x10, ___x9_7);
                                ___x9_8 = _mm512_fmadd_ps(___x11_8, ___x10, ___x9_8);
                                ___x9_9 = _mm512_fmadd_ps(___x11_9, ___x10, ___x9_9);
                                ___x9_10 = _mm512_fmadd_ps(___x11_10, ___x10, ___x9_10);
                                ___x9_11 = _mm512_fmadd_ps(___x11_11, ___x10, ___x9_11);
                                ___x9_12 = _mm512_fmadd_ps(___x11_12, ___x10, ___x9_12);
                                ___x9_13 = _mm512_fmadd_ps(___x11_13, ___x10, ___x9_13);
                                ___x9_14 = _mm512_fmadd_ps(___x11_14, ___x10, ___x9_14);
                                ___x9_15 = _mm512_fmadd_ps(___x11_15, ___x10, ___x9_15);
                                ___x9_16 = _mm512_fmadd_ps(___x11_16, ___x10, ___x9_16);
                                ___x9_17 = _mm512_fmadd_ps(___x11_17, ___x10, ___x9_17);
                                ___x9_18 = _mm512_fmadd_ps(___x11_18, ___x10, ___x9_18);
                                ___x9_19 = _mm512_fmadd_ps(___x11_19, ___x10, ___x9_19);
                                ___x9_20 = _mm512_fmadd_ps(___x11_20, ___x10, ___x9_20);
                                ___x9_21 = _mm512_fmadd_ps(___x11_21, ___x10, ___x9_21);
                                ___x9_22 = _mm512_fmadd_ps(___x11_22, ___x10, ___x9_22);
                                ___x9_23 = _mm512_fmadd_ps(___x11_23, ___x10, ___x9_23);
                                ___x9_24 = _mm512_fmadd_ps(___x11_24, ___x10, ___x9_24);
                                ___x9_25 = _mm512_fmadd_ps(___x11_25, ___x10, ___x9_25);
                                ___x9_26 = _mm512_fmadd_ps(___x11_26, ___x10, ___x9_26);
                                ___x9_27 = _mm512_fmadd_ps(___x11_27, ___x10, ___x9_27);
                            }
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 0) * 1)][0], ___x9_0);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 1) * 1)][0], ___x9_1);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 2) * 1)][0], ___x9_2);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 3) * 1)][0], ___x9_3);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 4) * 1)][0], ___x9_4);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 5) * 1)][0], ___x9_5);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 6) * 1)][0], ___x9_6);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 7) * 1)][0], ___x9_7);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 8) * 1)][0], ___x9_8);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 9) * 1)][0], ___x9_9);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 10) * 1)][0], ___x9_10);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 11) * 1)][0], ___x9_11);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 12) * 1)][0], ___x9_12);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 13) * 1)][0], ___x9_13);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 14) * 1)][0], ___x9_14);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 15) * 1)][0], ___x9_15);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 16) * 1)][0], ___x9_16);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 17) * 1)][0], ___x9_17);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 18) * 1)][0], ___x9_18);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 19) * 1)][0], ___x9_19);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 20) * 1)][0], ___x9_20);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 21) * 1)][0], ___x9_21);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 22) * 1)][0], ___x9_22);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 23) * 1)][0], ___x9_23);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 24) * 1)][0], ___x9_24);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 25) * 1)][0], ___x9_25);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 26) * 1)][0], ___x9_26);
                            _mm512_store_ps(& ensemble6grad_inputs[_neuron_index_0][i_outer][(j * 1 + _neuron_index_2 * 1)][(k * 1 + (0 + 27) * 1)][0], ___x9_27);
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel for collapse(2)
    for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
        for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 4; _neuron_index_1_outer += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 28; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 28; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        int _input_offset_1_outer = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) / 16;
                        int _input_offset_1_inner = (_neuron_index_1_outer * 16 + _neuron_index_1_inner) % 16;
                        int in_y = _neuron_index_2 * 2 - 1;
                        int _input_offset_2 = in_y;
                        int in_x = _neuron_index_3 * 2 - 1;
                        int _input_offset_3 = in_x;
                        long j = ensemble5mask_j[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        long k = ensemble5mask_k[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                        ensemble5grad_inputs[_neuron_index_0][_input_offset_1_outer][MIN(MAX(j + _input_offset_2, 0), 54)][MIN(MAX(k + _input_offset_3, 0), 54)][_input_offset_1_inner] += ensemble5grad[_neuron_index_0][_neuron_index_1_outer][(_neuron_index_2 + 2)][(_neuron_index_3 + 2)][_neuron_index_1_inner];
                    }
                }
            }
            for (int _neuron_index_2 = 0; _neuron_index_2 < 55; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 55; _neuron_index_3 += 1) {
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
void backward20(float* _ensemble3grad_bias, float* _ensemble2grad, float* _ensemble2grad_weights, float* _ensemble2inputs, float* _ensemble3grad) {
    float (* ensemble3grad)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble3grad;
    __assume_aligned(ensemble3grad, 64);
    float (* ensemble2inputs)[1][228][228][16] = (float (*)[1][228][228][16]) _ensemble2inputs;
    __assume_aligned(ensemble2inputs, 64);
    float (* ensemble2grad_weights)[1][11][11][16][16] = (float (*)[1][11][11][16][16]) _ensemble2grad_weights;
    __assume_aligned(ensemble2grad_weights, 64);
    float (* ensemble2grad)[4][55][55][16] = (float (*)[4][55][55][16]) _ensemble2grad;
    __assume_aligned(ensemble2grad, 64);
    float (* ensemble3grad_bias)[1][16] = (float (*)[1][16]) _ensemble3grad_bias;
    __assume_aligned(ensemble3grad_bias, 64);
    #pragma omp parallel for
    for (int _neuron_index_1_outer = 0; _neuron_index_1_outer < 4; _neuron_index_1_outer += 1) {
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 55; _neuron_index_2 += 1) {
                for (int _neuron_index_3 = 0; _neuron_index_3 < 55; _neuron_index_3 += 1) {
                    for (int _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                        ensemble3grad_bias[_neuron_index_1_outer][0][_neuron_index_1_inner] += ensemble3grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][_neuron_index_3][_neuron_index_1_inner];
                    }
                }
            }
        }
        for (int _neuron_index_0 = 0; _neuron_index_0 < 128; _neuron_index_0 += 1) {
            for (int _neuron_index_2 = 0; _neuron_index_2 < 55; _neuron_index_2 += 1) {
                int in_y = _neuron_index_2 * 4;
                int _input_offset_2 = in_y;
                for (int _neuron_index_3 = 0; _neuron_index_3 < 55; _neuron_index_3 += 11) {
                    int in_x_0 = (_neuron_index_3 + 0) * 4;
                    int in_x_1 = (_neuron_index_3 + 1) * 4;
                    int in_x_2 = (_neuron_index_3 + 2) * 4;
                    int in_x_3 = (_neuron_index_3 + 3) * 4;
                    int in_x_4 = (_neuron_index_3 + 4) * 4;
                    int in_x_5 = (_neuron_index_3 + 5) * 4;
                    int in_x_6 = (_neuron_index_3 + 6) * 4;
                    int in_x_7 = (_neuron_index_3 + 7) * 4;
                    int in_x_8 = (_neuron_index_3 + 8) * 4;
                    int in_x_9 = (_neuron_index_3 + 9) * 4;
                    int in_x_10 = (_neuron_index_3 + 10) * 4;
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
                    for (int j = 0; j < 11; j += 1) {
                        for (int k = 0; k < 11; k += 1) {
                            __m512 ___x4_0 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 0) * 4)][0]);
                            __m512 ___x4_1 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 1) * 4)][0]);
                            __m512 ___x4_2 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 2) * 4)][0]);
                            __m512 ___x4_3 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 3) * 4)][0]);
                            __m512 ___x4_4 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 4) * 4)][0]);
                            __m512 ___x4_5 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 5) * 4)][0]);
                            __m512 ___x4_6 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 6) * 4)][0]);
                            __m512 ___x4_7 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 7) * 4)][0]);
                            __m512 ___x4_8 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 8) * 4)][0]);
                            __m512 ___x4_9 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 9) * 4)][0]);
                            __m512 ___x4_10 = _mm512_load_ps(& ensemble2inputs[_neuron_index_0][0][(j * 1 + _neuron_index_2 * 4)][(k * 1 + (_neuron_index_3 + 10) * 4)][0]);
                            for (long _neuron_index_1_inner = 0; _neuron_index_1_inner < 16; _neuron_index_1_inner += 1) {
                                __m512 ___x3 = _mm512_load_ps(& ensemble2grad_weights[_neuron_index_1_outer][0][j][k][_neuron_index_1_inner][0]);
                                __m512 ___x5_0 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 0)][_neuron_index_1_inner]);
                                __m512 ___x5_1 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 1)][_neuron_index_1_inner]);
                                __m512 ___x5_2 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 2)][_neuron_index_1_inner]);
                                __m512 ___x5_3 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 3)][_neuron_index_1_inner]);
                                __m512 ___x5_4 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 4)][_neuron_index_1_inner]);
                                __m512 ___x5_5 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 5)][_neuron_index_1_inner]);
                                __m512 ___x5_6 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 6)][_neuron_index_1_inner]);
                                __m512 ___x5_7 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 7)][_neuron_index_1_inner]);
                                __m512 ___x5_8 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 8)][_neuron_index_1_inner]);
                                __m512 ___x5_9 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 9)][_neuron_index_1_inner]);
                                __m512 ___x5_10 = _mm512_set1_ps(ensemble2grad[_neuron_index_0][_neuron_index_1_outer][_neuron_index_2][(_neuron_index_3 + 10)][_neuron_index_1_inner]);
                                ___x3 = _mm512_fmadd_ps(___x5_0, ___x4_0, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_1, ___x4_1, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_2, ___x4_2, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_3, ___x4_3, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_4, ___x4_4, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_5, ___x4_5, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_6, ___x4_6, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_7, ___x4_7, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_8, ___x4_8, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_9, ___x4_9, ___x3);
                                ___x3 = _mm512_fmadd_ps(___x5_10, ___x4_10, ___x3);
                                _mm512_store_ps(& ensemble2grad_weights[_neuron_index_1_outer][0][j][k][_neuron_index_1_inner][0], ___x3);
                            }
                        }
                    }
                }
            }
        }
    }
};
