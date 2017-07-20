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
