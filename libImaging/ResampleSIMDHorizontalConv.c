
void
ImagingResampleHorizontalConvolution8u(UINT32 *lineOut, UINT32 *lineIn,
    int xsize, int *xbounds, INT16 *kk, int kmax, int coefs_precision)
{
    int xmin, xmax, xx, x;
    INT16 *k;

    for (xx = 0; xx < xsize; xx++) {
        __m128i sss;
        xmin = xbounds[xx * 2 + 0];
        xmax = xbounds[xx * 2 + 1];
        k = &kk[xx * kmax];
        x = 0;

// #if defined(__AVX2__)

//         __m256i sss256 = _mm256_set1_epi32((1 << (coefs_precision - 9 -2)) + xmax / 4);

//         for (; x < xmax - 3; x += 4) {
//             __m256i pix, mmk, mul;
//             __m256i source = _mm256_castsi128_si256(
//                 _mm_loadu_si128((__m128i *) &lineIn[x + xmin]));
//             __m256i ksource = _mm256_set1_epi64x(*(int64_t *) &k[x]);
//             source = _mm256_unpacklo_epi64(source, source);

//             pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
//             mmk = _mm256_shuffle_epi8(ksource, _mm256_set_epi8(
//                 3,2, 3,2, 3,2, 3,2, 1,0, 1,0, 1,0, 1,0,
//                 3,2, 3,2, 3,2, 3,2, 1,0, 1,0, 1,0, 1,0));
//             mul = _mm256_mulhi_epi16(_mm256_slli_epi16(pix, 7), mmk);
//             sss256 = _mm256_add_epi32(sss256, _mm256_cvtepi16_epi32(mul));
//             sss256 = _mm256_add_epi32(sss256, _mm256_cvtepi16_epi32(_mm256_srli_si128(mul, 8)));
//         }

//         sss = _mm_add_epi32(
//             _mm256_extractf128_si256(sss256, 0),
//             _mm256_extractf128_si256(sss256, 1)
//         );

// #else

        sss = _mm_set1_epi32((1 << (coefs_precision -1)) + xmax / 2);

        for (; x < xmax - 7; x += 8) {
            __m128i pix, mmk, source;
            __m128i ksource = _mm_loadu_si128((__m128i *) &k[x]);

            source = _mm_loadu_si128((__m128i *) &lineIn[x + xmin]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,11, -1,3, -1,10, -1,2, -1,9, -1,1, -1,8, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                5,4, 1,0, 5,4, 1,0, 5,4, 1,0, 5,4, 1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,15, -1,7, -1,14, -1,6, -1,13, -1,5, -1,12, -1,4));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                7,6, 3,2, 7,6, 3,2, 7,6, 3,2, 7,6, 3,2));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            source = _mm_loadu_si128((__m128i *) &lineIn[x + 4 + xmin]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,11, -1,3, -1,10, -1,2, -1,9, -1,1, -1,8, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                13,12, 9,8, 13,12, 9,8, 13,12, 9,8, 13,12, 9,8));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,15, -1,7, -1,14, -1,6, -1,13, -1,5, -1,12, -1,4));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                15,14, 11,10, 15,14, 11,10, 15,14, 11,10, 15,14, 11,10));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax - 3; x += 4) {
            __m128i pix, mmk;
            __m128i source = _mm_loadu_si128((__m128i *) &lineIn[x + xmin]);
            __m128i ksource = _mm_cvtsi64_si128(*(int64_t *) &k[x]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,11, -1,3, -1,10, -1,2, -1,9, -1,1, -1,8, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                5,4, 1,0, 5,4, 1,0, 5,4, 1,0, 5,4, 1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,15, -1,7, -1,14, -1,6, -1,13, -1,5, -1,12, -1,4));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                7,6, 3,2, 7,6, 3,2, 7,6, 3,2, 7,6, 3,2));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

        for (; x < xmax - 1; x += 2) {
            __m128i pix, mmk;
            __m128i source = _mm_cvtsi64_si128(*(int64_t *) &lineIn[x + xmin]);
            __m128i ksource = _mm_cvtsi32_si128(*(int *) &k[x]);

            pix = _mm_shuffle_epi8(source, _mm_set_epi8(
                -1,7, -1,3, -1,6, -1,2, -1,5, -1,1, -1,4, -1,0));
            mmk = _mm_shuffle_epi8(ksource, _mm_set_epi8(
                3,2, 1,0, 3,2, 1,0, 3,2, 1,0, 3,2, 1,0));
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }

// #endif

        for (; x < xmax; x ++) {
            __m128i pix = _mm_cvtepu8_epi32(*(__m128i *) &lineIn[x + xmin]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }
}
