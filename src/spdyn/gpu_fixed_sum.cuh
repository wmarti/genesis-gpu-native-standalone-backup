/*
 * gpu_fixed_sum.cuh : order-independent accumulation of doubles.
 *
 * A double atomicAdd rounds after every add, so the sum depends on the order
 * the adds land in. Here a sum is two 64-bit integer words and every add is
 * exact integer arithmetic, so the result is the same bits in any order:
 *
 *   w[0]  signed multiple of 2^-K           (each term: floor(v * 2^K))
 *   w[1]  unsigned multiple of 2^-(K+32)    (the rest, in [0, 2^32))
 *
 * v is first rounded to the nearest multiple of 2^-(K+32), so a term is kept
 * to 2^-(K+33) absolute (K = 24, the energy and virial slots: 6.9e-18), less
 * than one double atomicAdd rounding for any sum above ~0.1. The value is
 * rounded to double once, at the end. Range: |w[0] * 2^-K| < 2^(63-K).
 *
 * A term that is not finite or not below 2^(62-K) in magnitude sets bit 63
 * of w[1] (which the lo sum cannot reach), and the value then reads back as
 * NaN. Warps and blocks fold the lo carry into hi before their add, so w[1]
 * takes < 2^32 per add; value() reads the words back canonically.
 *
 * The two words of a sum are adjacent (w[2*i], w[2*i+1]), so both adds of a
 * term hit one 16-byte span.
 */
#ifndef GCN_FIXED_SUM_CUH
#define GCN_FIXED_SUM_CUH

#include <cmath>

namespace gcn_fx {

typedef unsigned long long word;

const int kEnergy = 24;   /* energy and virial slots               */

#ifdef __CUDACC__
/* The two words one term adds: *hi to w[0], *lo to w[1]. True when the term
 * cannot be represented (the words are then zero). The split is integer work
 * on the double's bits (FP64 issue is scarce on many GPUs). Encoded terms may
 * be summed as integers before they reach w: hi modulo 2^64, lo unnormalised
 * and far from 2^63. */
template <int K>
__device__ __forceinline__ bool encode(double v, word *hi_out, word *lo_out)
{
    *hi_out = 0;
    *lo_out = 0;
    const long long b = __double_as_longlong(v);
    const int ex = (int)((b >> 52) & 0x7ff);
    const int s = ex - 1075 + K + 32;         /* |v| = m * 2^(s - K - 32) */
    if (ex == 0x7ff || s > 41) return true;   /* not finite, or >= 2^(62-K) */
    if (s < -54) return false;                /* below half a unit, or zero */
    /* Selects, not branches: the lanes of a warp hold terms of every
     * magnitude, and a branch per exponent range would serialise them. */
    const word m  = ((word)b & ((1ull << 52) - 1)) | (1ull << 52);
    const int  rs = s < 0 ? -s : 0;
    const int  ls = s < 0 ? 0 : s;
    const word r  = (m + ((1ull << rs) >> 1)) >> rs;      /* to nearest */
    word hi = ls >= 32 ? r << (ls - 32) : r >> (32 - ls); /* |v| = (hi*2^32 */
    word lo = (r << ls) & 0xffffffffull;                  /*  + lo) 2^-(K+32) */
    if (b < 0) {                              /* -(hi*2^32 + lo) */
        hi = lo != 0 ? ~hi : 0 - hi;
        lo = lo != 0 ? 4294967296ull - lo : 0;
    }
    *hi_out = hi;
    *lo_out = lo;
    return false;
}

template <int K>
__device__ __forceinline__ void add(word *w, double v)
{
    word hi, lo;
    if (encode<K>(v, &hi, &lo)) {
        atomicOr(&w[1], 1ull << 63);
        return;
    }
    if (hi != 0) atomicAdd(&w[0], hi);
    if (lo != 0) atomicAdd(&w[1], lo);
}
#endif

/* Single-word sums, for fields that take many adds per step (forces, the PME
 * charge brick): a term is rounded to the nearest multiple of 2^-R and added
 * as one signed 64-bit integer, one atomic per term. Forces keep 2^-36
 * (1.5e-11 kcal/mol/A per term), the brick 2^-44 (charges below 3). A slot
 * takes far fewer than 2^9 terms a step, so a term below 2^(54-R) cannot
 * overflow the sum; a larger or non-finite term sets the translation unit's
 * overflow flag and every value the unit folds then reads back as NaN. */
const int kForce1 = 36;   /* force fields     */
const int kCharge = 44;   /* PME charge brick */

#ifdef __CUDACC__
static __device__ unsigned int overflow;

template <int R>
__device__ __forceinline__ void add1(word *w, double v)
{
    const double t = v * (double)(1ull << R);         /* exact scale */
    if (!(fabs(t) < 18014398509481984.0)) {          /* 2^54        */
        atomicOr(&overflow, 1u);
        return;
    }
    const long long q = __double2ll_rn(t);
    if (q != 0) atomicAdd(w, (word)q);
}

/* The same add for an FP32 term: v * 2^R is exact in either precision and
 * rounds to the same integer as add1(w, (double)v). */
template <int R>
__device__ __forceinline__ void add1(word *w, float v)
{
    const float t = v * (float)(1ull << R);           /* exact scale */
    if (!(fabsf(t) < 18014398509481984.0f)) {        /* 2^54        */
        atomicOr(&overflow, 1u);
        return;
    }
    const long long q = __float2ll_rn(t);
    if (q != 0) atomicAdd(w, (word)q);
}

/* value1 against another unit's overflow flag */
template <int R>
__device__ __forceinline__ double value1_of(word w, unsigned int flag)
{
    if (flag) return (double)NAN;
    return (double)(long long)w * (1.0 / (double)(1ull << R));
}

template <int R>
__device__ __forceinline__ double value1(word w)
{
    return value1_of<R>(w, overflow);
}

/* FP32 variant for the PME charge brick of nonbond_precision = MIXED: a term
 * is rounded to the nearest multiple of 2^-R (R = kCharge32) and added as one
 * signed 32-bit integer. The sum wraps modulo 2^32, so it is exact and order
 * independent while the final value lies in (-2^(31-R), 2^(31-R)); a term
 * outside it sets the overflow flag. */
const int kCharge32 = 24;

template <int R>
__device__ __forceinline__ void add1(unsigned int *w, float v)
{
    const float t = v * (float)(1u << R);             /* exact scale */
    if (!(fabsf(t) < 2147483648.0f)) {               /* 2^31        */
        atomicOr(&overflow, 1u);
        return;
    }
    const int q = __float2int_rn(t);
    if (q != 0) atomicAdd(w, (unsigned int)q);
}

template <int R>
__device__ __forceinline__ double value1(unsigned int w)
{
    if (overflow) return (double)NAN;
    return (double)(int)w * (1.0 / (double)(1u << R));
}

/* (float)value1<R>(w) bit for bit, without FP64 arithmetic. */
template <int R>
__device__ __forceinline__ float value1f(unsigned int w)
{
    if (overflow) return NAN;
    return (float)(int)w * (1.0f / (float)(1u << R));
}
#endif

/* The sum as a double: rounded once while |w[0]| < 2^53 (any force), else
 * with the ~1e-16 relative split of w[0]. */
template <int K>
#ifdef __CUDACC__
__host__ __device__
#endif
inline double value(word w0, word w1)
{
    if (w1 >> 63) return (double)NAN;
    w0 += w1 >> 32;                  /* canonical: the same (w0, w1) for */
    w1 &= 0xffffffffull;             /* every split of one sum           */
    const long long hi = (long long)w0;
    const double s = 1.0 / (double)(1ull << K);
    const double l = (double)w1 * (s / 4294967296.0);     /* exact below 2^53 */
    if (hi > -(1ll << 53) && hi < (1ll << 53))
        return fma((double)hi, s, l);                     /* one rounding */
    const double h = (double)hi;
    const long long r = hi - (long long)h;
    return h * s + ((double)r * s + l);
}

}  /* namespace gcn_fx */

#endif
