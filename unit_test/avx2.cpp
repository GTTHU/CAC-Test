#include <stdio.h>
#include <memory.h>
#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>


typedef __int8                     s8;
typedef unsigned __int8            u8;
typedef __int16                    s16;
typedef unsigned __int16           u16;
typedef __int32                    s32;
typedef unsigned __int32           u32;
typedef __int64                    s64;
typedef unsigned __int64           u64;

typedef s16                        pel; /* pixel type */
typedef s32                        double_pel; /* pixel type */

#if 0
#define AVX_SAD_16B_4PEL(src1, src2, s00, s01, s02, sum0) \
    s00 = _mm256_loadu_si256((__m256i*)(src1)); \
    s01 = _mm256_loadu_si256((__m256i*)(src2));\
    s00 = _mm256_sub_epi16(s00, s01); \
    s00 = _mm256_abs_epi16(s00); \
	s02 = _mm256_castsi256_si128(s00); \
	s00 = _mm256_cvtepi16_epi32(s02); \
	\
	sum0 = _mm256_add_epi32(sum0, s00); \

#define AVX_SAD_16B_8PEL(src1, src2, s00, s01, s02, sum0) \
    s00 = _mm256_loadu_si256((__m256i*)(src1)); \
    s01 = _mm256_loadu_si256((__m256i*)(src2)); \
    s00 = _mm256_sub_epi16(s00, s01); \
    s01 = _mm256_abs_epi16(s00); \
	s00 = _mm256_hadd_epi16(s01, s01); \
	s02 = _mm256_castsi256_si128(s00); \
	s00 = _mm256_cvtepi16_epi32(s02); \
	sum0 = _mm256_add_epi32(sum0, s00); \

#define AVX_SAD_16B_16PEL(src1, src2, s00, s01, sum0) \
    s00 = _mm256_loadu_si256((__m256i*)(src1)); \
    s01 = _mm256_loadu_si256((__m256i*)(src2)); \
    s00 = _mm256_sub_epi16(s00, s01); \
    s00 = _mm256_abs_epi16(s00); \
	s00 = _mm256_hadd_epi16(s00, s00); \
	sum0 = _mm256_add_epi16(sum0, s00); \

int sad_16b_w4_avx2(int w, int h, void * src1, void * src2, int s_src1, int s_src2, int bit_depth)
{
	int sad = 0;
	__m256i sum0 = _mm256_setzero_si256();
	__m256i s00, s01;
	__m128i s02;
	s16 * s1 = (s16 *)src1;
	s16 * s2 = (s16 *)src2;

	while (h--) {
		AVX_SAD_16B_4PEL(s1, s2, s00, s01, s02, sum0);

		s1 += s_src1;
		s2 += s_src2;
	}

	sad += _mm256_extract_epi32(sum0, 0);
	sad += _mm256_extract_epi32(sum0, 1);
	sad += _mm256_extract_epi32(sum0, 2);
	sad += _mm256_extract_epi32(sum0, 3);

	return sad >> (bit_depth - 8);
}

int sad_16b_w8_avx2(int w, int h, void * src1, void * src2, int s_src1, int s_src2, int bit_depth)
{
	int sad = 0;
	__m256i sum0 = _mm256_setzero_si256();
	__m256i s00, s01;
	__m128i s02;
	s16 * s1 = (s16 *)src1;
	s16 * s2 = (s16 *)src2;

	while (h--) {
		AVX_SAD_16B_8PEL(s1, s2, s00, s01, s02, sum0);

		s1 += s_src1;
		s2 += s_src2;
	}

	sad += _mm256_extract_epi32(sum0, 0);
	sad += _mm256_extract_epi32(sum0, 1);
	sad += _mm256_extract_epi32(sum0, 2);
	sad += _mm256_extract_epi32(sum0, 3);

	return sad >> (bit_depth - 8);
}

int sad_16b_w16_avx2(int w, int h, void * src1, void * src2, int s_src1, int s_src2, int bit_depth)
{
	int sad = 0;
	__m256i sum0 = _mm256_setzero_si256();
	__m256i s00, s01;
	s16 * s1 = (s16 *)src1;
	s16 * s2 = (s16 *)src2;

	while (h--)
	{
		AVX_SAD_16B_16PEL(s1, s2, s00, s01, sum0);
		s1 += s_src1;
		s2 += s_src2;
	}
	sad += _mm256_extract_epi16(sum0, 0);
	sad += _mm256_extract_epi16(sum0, 1);
	sad += _mm256_extract_epi16(sum0, 2);
	sad += _mm256_extract_epi16(sum0, 3);
	sad += _mm256_extract_epi16(sum0, 4);
	sad += _mm256_extract_epi16(sum0, 5);
	sad += _mm256_extract_epi16(sum0, 6);
	sad += _mm256_extract_epi16(sum0, 7);
	return (sad >> (bit_depth - 8));
}

int sad_16b_w32_avx2(int w, int h, void * src1, void * src2, int s_src1, int s_src2, int bit_depth)
{
	int sad = 0;
	__m256i sum0 = _mm256_setzero_si256();
	__m256i s00, s01;
	__m128i s02;
	s16 * s1 = (s16 *)src1;
	s16 * s2 = (s16 *)src2;

	while (h--)
	{
		AVX_SAD_16B_8PEL(s1, s2, s00, s01, s02, sum0);
		AVX_SAD_16B_8PEL(s1 + 8, s2 + 8, s00, s01, s02, sum0);
		AVX_SAD_16B_8PEL(s1 + 16, s2 + 16, s00, s01, s02, sum0);
		AVX_SAD_16B_8PEL(s1 + 24, s2 + 24, s00, s01, s02, sum0);
		s1 += s_src1;
		s2 += s_src2;
	}
	sad += _mm256_extract_epi32(sum0, 0);
	sad += _mm256_extract_epi32(sum0, 1);
	sad += _mm256_extract_epi32(sum0, 2);
	sad += _mm256_extract_epi32(sum0, 3);
	return (sad >> (bit_depth - 8));
	return 0;
}

int sad_16b_w8n_avx2(int w, int h, void * src1, void * src2, int s_src1, int s_src2, int bit_depth)
{
	int sad = 0;
	__m256i sum0 = _mm256_setzero_si256();
	__m256i s00, s01;
	__m128i s02;
	s16 * s1 = (s16 *)src1;
	s16 * s2 = (s16 *)src2;

	int w8 = w >> 3;
	while (h--)
	{
		for (int w_idx = 0; w_idx < w8; w_idx++)
		{
			int w_offset = w_idx << 3;
			AVX_SAD_16B_8PEL(s1 + w_offset, s2 + w_offset, s00, s01, s02, sum0);
		}
		s1 += s_src1;
		s2 += s_src2;
	}
	sad += _mm256_extract_epi32(sum0, 0);
	sad += _mm256_extract_epi32(sum0, 1);
	sad += _mm256_extract_epi32(sum0, 2);
	sad += _mm256_extract_epi32(sum0, 3);
	return (sad >> (bit_depth - 8));
}
#endif