#include "Imaging.h"

#include <math.h>
#include <emmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif


#define ROUND_UP(f) ((int) ((f) >= 0.0 ? (f) + 0.5F : (f) - 0.5F))


struct filter {
    double (*filter)(double x);
    double support;
};

static inline double sinc_filter(double x)
{
    if (x == 0.0)
        return 1.0;
    x = x * M_PI;
    return sin(x) / x;
}

static inline double lanczos_filter(double x)
{
    /* truncated sinc */
    if (-3.0 <= x && x < 3.0)
        return sinc_filter(x) * sinc_filter(x/3);
    return 0.0;
}

static inline double bilinear_filter(double x)
{
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return 1.0-x;
    return 0.0;
}

static inline double bicubic_filter(double x)
{
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm */
#define a -0.5
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x*x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
#undef a
}

static struct filter LANCZOS = { lanczos_filter, 3.0 };
static struct filter BILINEAR = { bilinear_filter, 1.0 };
static struct filter BICUBIC = { bicubic_filter, 2.0 };



/* The number of bits available in the accumulator.
   Filters can have negative areas. We need at least 1 bit for that
   and 1 extra bit for overflow. 8 bits is the result. */
#define MAX_PRECISION_BITS (32 - 8 - 2)

/* We use signed INT16 type to store coefficients. */
#define MAX_COEFS_PRECISION (16 - 1)

UINT8 lookups[512] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

static inline UINT8 clip8(int in, int precision)
{
    return lookups[(in >> precision) + 128];
}


int
precompute_coeffs(int inSize, int outSize, struct filter *filterp,
                  int **xboundsp, double **kkp) {
    double support, scale, filterscale;
    double center, ww, ss;
    int xx, x, kmax, xmin, xmax;
    int *xbounds;
    double *kk, *k;

    /* prepare for horizontal stretch */
    filterscale = scale = (double) inSize / outSize;
    if (filterscale < 1.0) {
        filterscale = 1.0;
    }

    /* determine support size (length of resampling filter) */
    support = filterp->support * filterscale;

    /* maximum number of coeffs */
    kmax = (int) ceil(support) * 2 + 1;

    // check for overflow
    if (outSize > INT_MAX / (kmax * sizeof(double)))
        return 0;

    /* coefficient buffer */
    /* malloc check ok, overflow checked above */
    kk = malloc(outSize * kmax * sizeof(double));
    if ( ! kk)
        return 0;

    /* malloc check ok, kmax*sizeof(double) > 2*sizeof(int) */
    xbounds = malloc(outSize * 2 * sizeof(int));
    if ( ! xbounds) {
        free(kk);
        return 0;
    }

    for (xx = 0; xx < outSize; xx++) {
        center = (xx + 0.5) * scale;
        ww = 0.0;
        ss = 1.0 / filterscale;
        xmin = (int) floor(center - support);
        if (xmin < 0)
            xmin = 0;
        xmax = (int) ceil(center + support);
        if (xmax > inSize)
            xmax = inSize;
        xmax -= xmin;
        k = &kk[xx * kmax];
        for (x = 0; x < xmax; x++) {
            double w = filterp->filter((x + xmin - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        for (x = 0; x < xmax; x++) {
            if (ww != 0.0)
                k[x] /= ww;
        }
        // Remaining values should stay empty if they are used despite of xmax.
        for (; x < kmax; x++) {
            k[x] = 0;
        }
        xbounds[xx * 2 + 0] = xmin;
        xbounds[xx * 2 + 1] = xmax;
    }
    *xboundsp = xbounds;
    *kkp = kk;
    return kmax;
}


int
normalize_coeffs(int inSize, int kmax, double *prekk, INT16 **kkp)
{
    int x;
    int coefs_precision;
    INT16 *kk;
    double maxkk;

    kk = malloc(inSize * kmax * sizeof(INT16));
    if ( ! kk) {
        return 0;
    }

    maxkk = prekk[0];
    for (x = 0; x < inSize * kmax; x++) {
        if (maxkk < prekk[x]) {
            maxkk = prekk[x];
        }
    }

    coefs_precision = 0;
    while (maxkk < (1 << (MAX_COEFS_PRECISION-1))) {
        maxkk *= 2;
        coefs_precision += 1;
    };

    if (coefs_precision > MAX_PRECISION_BITS) {
        coefs_precision = MAX_PRECISION_BITS;
    }

    for (x = 0; x < inSize * kmax; x++) {
        if (prekk[x] > 0) {
            kk[x] = (int) (0.5 + prekk[x] * (1 << coefs_precision));
        } else {
            kk[x] = (int) (-0.5 + prekk[x] * (1 << coefs_precision));
        }
    }

    *kkp = kk;
    return coefs_precision;
}


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

void
ImagingResampleVerticalConvolution8u(UINT32 *lineOut, Imaging imIn,
    int xmin, int xmax, INT16 *k, int coefs_precision)
{
    int x;
    int xx = 0;
    int xsize = imIn->xsize;

    __m128i initial = _mm_set1_epi32((1 << (coefs_precision -1)) + xmax / 2);

#if defined(__AVX2__)

    __m256i initial_256 = _mm256_set1_epi32((1 << (coefs_precision -1)) + xmax / 2);

    for (; xx < xsize - 7; xx += 8) {
        __m256i sss0 = initial_256;
        __m256i sss1 = initial_256;
        __m256i sss2 = initial_256;
        __m256i sss3 = initial_256;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m256i source, source1, source2;
            __m256i pix, mmk, mmk1;
            mmk = _mm256_set1_epi32(k[x]);
            mmk1 = _mm256_set1_epi32(k[x + 1]);
            mmk = _mm256_unpacklo_epi16(
                _mm256_packs_epi32(mmk, mmk),
                _mm256_packs_epi32(mmk1, mmk1));
            
            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &imIn->image32[x + xmin][xx]);
            source2 = _mm256_loadu_si256(  // bottom line
                (__m256i *) &imIn->image32[x + 1 + xmin][xx]);

            source = _mm256_unpacklo_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, source2);
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m256i source, source1, pix, mmk;
            mmk = _mm256_set1_epi32(k[x]);
            
            source1 = _mm256_loadu_si256(  // top line
                (__m256i *) &imIn->image32[x + xmin][xx]);
            
            source = _mm256_unpacklo_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss0 = _mm256_add_epi32(sss0, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss1 = _mm256_add_epi32(sss1, _mm256_madd_epi16(pix, mmk));

            source = _mm256_unpackhi_epi8(source1, _mm256_setzero_si256());
            pix = _mm256_unpacklo_epi8(source, _mm256_setzero_si256());
            sss2 = _mm256_add_epi32(sss2, _mm256_madd_epi16(pix, mmk));
            pix = _mm256_unpackhi_epi8(source, _mm256_setzero_si256());
            sss3 = _mm256_add_epi32(sss3, _mm256_madd_epi16(pix, mmk));
        }
        sss0 = _mm256_srai_epi32(sss0, coefs_precision);
        sss1 = _mm256_srai_epi32(sss1, coefs_precision);
        sss2 = _mm256_srai_epi32(sss2, coefs_precision);
        sss3 = _mm256_srai_epi32(sss3, coefs_precision);

        sss0 = _mm256_packs_epi32(sss0, sss1);
        sss2 = _mm256_packs_epi32(sss2, sss3);
        sss0 = _mm256_packus_epi16(sss0, sss2);
        _mm256_storeu_si256((__m256i *) &lineOut[xx], sss0);
    }

#else

    for (; xx < xsize - 7; xx += 8) {
        __m128i sss0 = initial;
        __m128i sss1 = initial;
        __m128i sss2 = initial;
        __m128i sss3 = initial;
        __m128i sss4 = initial;
        __m128i sss5 = initial;
        __m128i sss6 = initial;
        __m128i sss7 = initial;
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk, mmk1;
            mmk = _mm_set1_epi32(k[x]);
            mmk1 = _mm_set1_epi32(k[x + 1]);
            mmk = _mm_unpacklo_epi16(
                _mm_packs_epi32(mmk, mmk),
                _mm_packs_epi32(mmk1, mmk1));
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &imIn->image32[x + 1 + xmin][xx]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx + 4]);
            source2 = _mm_loadu_si128(  // bottom line
                (__m128i *) &imIn->image32[x + 1 + xmin][xx + 4]);

            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = _mm_add_epi32(sss4, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = _mm_add_epi32(sss5, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = _mm_add_epi32(sss6, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = _mm_add_epi32(sss7, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            
            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss2 = _mm_add_epi32(sss2, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss3 = _mm_add_epi32(sss3, _mm_madd_epi16(pix, mmk));

            source1 = _mm_loadu_si128(  // top line
                (__m128i *) &imIn->image32[x + xmin][xx + 4]);

            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss4 = _mm_add_epi32(sss4, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss5 = _mm_add_epi32(sss5, _mm_madd_epi16(pix, mmk));

            source = _mm_unpackhi_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss6 = _mm_add_epi32(sss6, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss7 = _mm_add_epi32(sss7, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);
        sss2 = _mm_srai_epi32(sss2, coefs_precision);
        sss3 = _mm_srai_epi32(sss3, coefs_precision);
        sss4 = _mm_srai_epi32(sss4, coefs_precision);
        sss5 = _mm_srai_epi32(sss5, coefs_precision);
        sss6 = _mm_srai_epi32(sss6, coefs_precision);
        sss7 = _mm_srai_epi32(sss7, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss2 = _mm_packs_epi32(sss2, sss3);
        sss0 = _mm_packus_epi16(sss0, sss2);
        _mm_storeu_si128((__m128i *) &lineOut[xx], sss0);
        sss4 = _mm_packs_epi32(sss4, sss5);
        sss6 = _mm_packs_epi32(sss6, sss7);
        sss4 = _mm_packus_epi16(sss4, sss6);
        _mm_storeu_si128((__m128i *) &lineOut[xx + 4], sss4);
    }

    for (; xx < xsize - 1; xx += 2) {   
        __m128i sss0 = initial;  // left row
        __m128i sss1 = initial;  // right row
        x = 0;
        for (; x < xmax - 1; x += 2) {
            __m128i source, source1, source2;
            __m128i pix, mmk, mmk1;
            mmk = _mm_set1_epi32(k[x]);
            mmk1 = _mm_set1_epi32(k[x + 1]);
            mmk = _mm_unpacklo_epi16(
                _mm_packs_epi32(mmk, mmk),
                _mm_packs_epi32(mmk1, mmk1));

            source1 = _mm_cvtsi64_si128(  // top line
                *(int64_t *) &imIn->image32[x + xmin][xx]);
            source2 = _mm_cvtsi64_si128(  // bottom line
                *(int64_t *) &imIn->image32[x + 1 + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, source2);
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        for (; x < xmax; x += 1) {
            __m128i source, source1, pix, mmk;
            mmk = _mm_set1_epi32(k[x]);
            
            source1 = _mm_cvtsi64_si128(  // top line
                *(int64_t *) &imIn->image32[x + xmin][xx]);
            
            source = _mm_unpacklo_epi8(source1, _mm_setzero_si128());
            pix = _mm_unpacklo_epi8(source, _mm_setzero_si128());
            sss0 = _mm_add_epi32(sss0, _mm_madd_epi16(pix, mmk));
            pix = _mm_unpackhi_epi8(source, _mm_setzero_si128());
            sss1 = _mm_add_epi32(sss1, _mm_madd_epi16(pix, mmk));
        }
        sss0 = _mm_srai_epi32(sss0, coefs_precision);
        sss1 = _mm_srai_epi32(sss1, coefs_precision);

        sss0 = _mm_packs_epi32(sss0, sss1);
        sss0 = _mm_packus_epi16(sss0, sss0);
        *(int64_t *) &lineOut[xx] = _mm_cvtsi128_si64x(sss0);
    }

#endif

    for (; xx < xsize; xx++) {
        __m128i sss = initial;
        for (x = 0; x < xmax; x++) {
            __m128i pix = _mm_cvtepu8_epi32(*(__m128i *) &imIn->image32[x + xmin][xx]);
            __m128i mmk = _mm_set1_epi32(k[x]);
            sss = _mm_add_epi32(sss, _mm_madd_epi16(pix, mmk));
        }
        sss = _mm_srai_epi32(sss, coefs_precision);
        sss = _mm_packs_epi32(sss, sss);
        lineOut[xx] = _mm_cvtsi128_si32(_mm_packus_epi16(sss, sss));
    }
}



Imaging
ImagingResampleHorizontal_8bpc(Imaging imIn, int xsize, struct filter *filterp)
{
    ImagingSectionCookie cookie;
    Imaging imOut;
    int ss0;
    int xx, yy, x, kmax, xmin, xmax;
    int *xbounds;
    INT16 *k, *kk;
    double *prekk;
    int coefs_precision;

    kmax = precompute_coeffs(imIn->xsize, xsize, filterp, &xbounds, &prekk);
    if ( ! kmax) {
        return (Imaging) ImagingError_MemoryError();
    }

    coefs_precision = normalize_coeffs(xsize, kmax, prekk, &kk);
    free(prekk);    
    if ( ! coefs_precision) {
        free(xbounds);
        return (Imaging) ImagingError_MemoryError();
    }

    imOut = ImagingNew(imIn->mode, xsize, imIn->ysize);
    if ( ! imOut) {
        free(kk);
        free(xbounds);
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    if (imIn->image8) {
        for (yy = 0; yy < imOut->ysize; yy++) {
            for (xx = 0; xx < xsize; xx++) {
                xmin = xbounds[xx * 2 + 0];
                xmax = xbounds[xx * 2 + 1];
                k = &kk[xx * kmax];
                ss0 = 1 << (coefs_precision -1);
                for (x = 0; x < xmax; x++)
                    ss0 += ((UINT8) imIn->image8[yy][x + xmin]) * k[x];
                imOut->image8[yy][xx] = clip8(ss0, coefs_precision);
            }
        }
    } else if (imIn->type == IMAGING_TYPE_UINT8) {
        for (yy = 0; yy < imOut->ysize; yy++) {
            ImagingResampleHorizontalConvolution8u(
                (UINT32 *) imOut->image32[yy],
                (UINT32 *) imIn->image32[yy],
                xsize, xbounds, kk, kmax,
                coefs_precision
            );
        }
    }

    ImagingSectionLeave(&cookie);
    free(kk);
    free(xbounds);
    return imOut;
}


Imaging
ImagingResampleVertical_8bpc(Imaging imIn, int ysize, struct filter *filterp)
{
    ImagingSectionCookie cookie;
    Imaging imOut;
    int ss0;
    int xx, yy, y, kmax, ymin, ymax;
    int *xbounds;
    INT16 *k, *kk;
    double *prekk;
    int coefs_precision;

    kmax = precompute_coeffs(imIn->ysize, ysize, filterp, &xbounds, &prekk);
    if ( ! kmax) {
        return (Imaging) ImagingError_MemoryError();
    }
    
    coefs_precision = normalize_coeffs(ysize, kmax, prekk, &kk);
    free(prekk);    
    if ( ! coefs_precision) {
        free(xbounds);
        return (Imaging) ImagingError_MemoryError();
    }

    imOut = ImagingNew(imIn->mode, imIn->xsize, ysize);
    if ( ! imOut) {
        free(kk);
        free(xbounds);
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    if (imIn->image8) {
        for (yy = 0; yy < ysize; yy++) {
            k = &kk[yy * kmax];
            ymin = xbounds[yy * 2 + 0];
            ymax = xbounds[yy * 2 + 1];
            for (xx = 0; xx < imOut->xsize; xx++) {
                ss0 = 1 << (coefs_precision -1);
                for (y = 0; y < ymax; y++)
                    ss0 += ((UINT8) imIn->image8[y + ymin][xx]) * k[y];
                imOut->image8[yy][xx] = clip8(ss0, coefs_precision);
            }
        }
    } else if (imIn->type == IMAGING_TYPE_UINT8) {
        for (yy = 0; yy < ysize; yy++) {
            k = &kk[yy * kmax];
            ymin = xbounds[yy * 2 + 0];
            ymax = xbounds[yy * 2 + 1];
            ImagingResampleVerticalConvolution8u(
                (UINT32 *) imOut->image32[yy], imIn,
                ymin, ymax, k, coefs_precision
            );
        }
    }

    ImagingSectionLeave(&cookie);
    free(kk);
    free(xbounds);
    return imOut;
}


Imaging
ImagingResampleHorizontal_32bpc(Imaging imIn, int xsize, struct filter *filterp)
{
    ImagingSectionCookie cookie;
    Imaging imOut;
    double ss;
    int xx, yy, x, kmax, xmin, xmax;
    int *xbounds;
    double *k, *kk;

    kmax = precompute_coeffs(imIn->xsize, xsize, filterp, &xbounds, &kk);
    if ( ! kmax) {
        return (Imaging) ImagingError_MemoryError();
    }

    imOut = ImagingNew(imIn->mode, xsize, imIn->ysize);
    if ( ! imOut) {
        free(kk);
        free(xbounds);
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    switch(imIn->type) {
        case IMAGING_TYPE_INT32:
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < xsize; xx++) {
                    xmin = xbounds[xx * 2 + 0];
                    xmax = xbounds[xx * 2 + 1];
                    k = &kk[xx * kmax];
                    ss = 0.0;
                    for (x = 0; x < xmax; x++)
                        ss += IMAGING_PIXEL_I(imIn, x + xmin, yy) * k[x];
                    IMAGING_PIXEL_I(imOut, xx, yy) = ROUND_UP(ss);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (yy = 0; yy < imOut->ysize; yy++) {
                for (xx = 0; xx < xsize; xx++) {
                    xmin = xbounds[xx * 2 + 0];
                    xmax = xbounds[xx * 2 + 1];
                    k = &kk[xx * kmax];
                    ss = 0.0;
                    for (x = 0; x < xmax; x++)
                        ss += IMAGING_PIXEL_F(imIn, x + xmin, yy) * k[x];
                    IMAGING_PIXEL_F(imOut, xx, yy) = ss;
                }
            }
            break;
    }

    ImagingSectionLeave(&cookie);
    free(kk);
    free(xbounds);
    return imOut;
}


Imaging
ImagingResampleVertical_32bpc(Imaging imIn, int ysize, struct filter *filterp)
{
    ImagingSectionCookie cookie;
    Imaging imOut;
    double ss;
    int xx, yy, y, kmax, ymin, ymax;
    int *xbounds;
    double *k, *kk;

    kmax = precompute_coeffs(imIn->ysize, ysize, filterp, &xbounds, &kk);
    if ( ! kmax) {
        return (Imaging) ImagingError_MemoryError();
    }

    imOut = ImagingNew(imIn->mode, imIn->xsize, ysize);
    if ( ! imOut) {
        free(kk);
        free(xbounds);
        return NULL;
    }

    ImagingSectionEnter(&cookie);
    switch(imIn->type) {
        case IMAGING_TYPE_INT32:
            for (yy = 0; yy < ysize; yy++) {
                ymin = xbounds[yy * 2 + 0];
                ymax = xbounds[yy * 2 + 1];
                k = &kk[yy * kmax];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    ss = 0.0;
                    for (y = 0; y < ymax; y++)
                        ss += IMAGING_PIXEL_I(imIn, xx, y + ymin) * k[y];
                    IMAGING_PIXEL_I(imOut, xx, yy) = ROUND_UP(ss);
                }
            }
            break;

        case IMAGING_TYPE_FLOAT32:
            for (yy = 0; yy < ysize; yy++) {
                ymin = xbounds[yy * 2 + 0];
                ymax = xbounds[yy * 2 + 1];
                k = &kk[yy * kmax];
                for (xx = 0; xx < imOut->xsize; xx++) {
                    ss = 0.0;
                    for (y = 0; y < ymax; y++)
                        ss += IMAGING_PIXEL_F(imIn, xx, y + ymin) * k[y];
                    IMAGING_PIXEL_F(imOut, xx, yy) = ss;
                }
            }
            break;
    }

    ImagingSectionLeave(&cookie);
    free(kk);
    free(xbounds);
    return imOut;
}


Imaging
ImagingResample(Imaging imIn, int xsize, int ysize, int filter)
{
    Imaging imTemp = NULL;
    Imaging imOut = NULL;
    struct filter *filterp;
    Imaging (*ResampleHorizontal)(Imaging imIn, int xsize, struct filter *filterp);
    Imaging (*ResampleVertical)(Imaging imIn, int xsize, struct filter *filterp);

    if (strcmp(imIn->mode, "P") == 0 || strcmp(imIn->mode, "1") == 0)
        return (Imaging) ImagingError_ModeError();

    if (imIn->type == IMAGING_TYPE_SPECIAL) {
        return (Imaging) ImagingError_ModeError();
    } else if (imIn->image8) {
        ResampleHorizontal = ImagingResampleHorizontal_8bpc;
        ResampleVertical = ImagingResampleVertical_8bpc;
    } else {
        switch(imIn->type) {
            case IMAGING_TYPE_UINT8:
                ResampleHorizontal = ImagingResampleHorizontal_8bpc;
                ResampleVertical = ImagingResampleVertical_8bpc;
                break;
            case IMAGING_TYPE_INT32:
            case IMAGING_TYPE_FLOAT32:
                ResampleHorizontal = ImagingResampleHorizontal_32bpc;
                ResampleVertical = ImagingResampleVertical_32bpc;
                break;
            default:
                return (Imaging) ImagingError_ModeError();
        }
    }

    /* check filter */
    switch (filter) {
    case IMAGING_TRANSFORM_LANCZOS:
        filterp = &LANCZOS;
        break;
    case IMAGING_TRANSFORM_BILINEAR:
        filterp = &BILINEAR;
        break;
    case IMAGING_TRANSFORM_BICUBIC:
        filterp = &BICUBIC;
        break;
    default:
        return (Imaging) ImagingError_ValueError(
            "unsupported resampling filter"
            );
    }

    /* two-pass resize, first pass */
    if (imIn->xsize != xsize) {
        imTemp = ResampleHorizontal(imIn, xsize, filterp);
        if ( ! imTemp)
            return NULL;
        imOut = imIn = imTemp;
    }

    /* second pass */
    if (imIn->ysize != ysize) {
        /* imIn can be the original image or horizontally resampled one */
        imOut = ResampleVertical(imIn, ysize, filterp);
        /* it's safe to call ImagingDelete with empty value
           if there was no previous step. */
        ImagingDelete(imTemp);
        if ( ! imOut)
            return NULL;
    }

    /* none of the previous steps are performed, copying */
    if ( ! imOut) {
        imOut = ImagingCopy(imIn);
    }

    return imOut;
}
