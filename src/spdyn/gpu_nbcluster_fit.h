/* gpu_nbcluster_fit.h : parameters of the analytic pair term and their
 * identification from the PME table (host code, no CUDA).  Private to
 * gpu_nbcluster.cu. */

#ifndef GPU_NBCLUSTER_FIT_H
#define GPU_NBCLUSTER_FIT_H

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <utility>
#include <vector>

namespace gcn {

/* Terms of the Ewald correction polynomials: kNpShort when (beta rc)^2 is
 * small enough for FP32 accuracy, else kNpLong. */
const int kNpShort = 12;
const int kNpLong = 18;

struct nbc_par {
    float rc2;      /* evaluation radius^2: the table's support, or the
                       cutoff (analytic)                                  */
    float cdf;      /* table: s = cdf / r2                                */
    float lmax;     /* table: the last record                             */
    int   erow;     /* table: energy records start at 2*erow              */
    int   ncls1;    /* LJ classes + the null class of fillers             */
    /* analytic: Ewald beta, t = r2*tA - 1 on [-1, 1] over [0, cutoff],
       the CHARMM switch ron2 -> roff2 (S = t1^2 (sw_a r2 + sw_b),
       dS/(r dr) = t1 (sw_d r2 + sw_e), t1 = roff2 - r2) and the
       correction polynomials, the force one times beta^3 */
    float beta, tA;
    float ron2, roff2, sw_a, sw_b, sw_d, sw_e;
    float pf[kNpLong], pe[kNpLong];
    int np;         /* terms used: 0 (table), kNpShort or kNpLong */
    /* analytic LJ form: CHARMM's potential switch (sw_*) or its force
       switch (fs_*): for r2 > ron2 the r^-12 term is k12 (r^-6 - roff^-6)^2
       and the r^-6 term k6 (r^-3 - roff^-3)^2, below it r^-12 - A12 and
       r^-6 - A6; fs = {k12, k12 roff^-6, k12 roff^-12, A12,
       k6, k6 roff^-3, k6 roff^-6, A6} */
    int fsw;
    float fs[8];
};

inline double ew_g(double z)   /* (erf x - 2x e^-z / sqrt(pi)) / x^3, z = x^2 */
{
    if (z < 6) {
        double s = 0, term = 1;
        for (int n = 1; n < 80; ++n) {
            term = n == 1 ? 1.0 : -term * z / n;
            s += term * 2.0 * n / (2.0 * n + 1.0);
        }
        return s * 2.0 / std::sqrt(M_PI);
    }
    const double x = std::sqrt(z);
    return (std::erf(x) - 2 * x * std::exp(-z) / std::sqrt(M_PI)) / (z * x);
}
inline double ew_h(double z)   /* erf(x) / x */
{
    if (z < 1e-3) {
        double s = 0, term = 1;
        for (int n = 0; n < 80; ++n) { if (n) term = -term * z / n; s += term / (2.0 * n + 1.0); }
        return s * 2.0 / std::sqrt(M_PI);
    }
    const double x = std::sqrt(z);
    return std::erf(x) / x;
}
/* monomial coefficients in t on [-1, 1] of the Chebyshev interpolant of
 * scale * f on z in [0, zmax], z = (t + 1) zmax / 2 */
inline void chebfit(double (*f)(double), double zmax, int n, double scale, float *out)
{
    std::vector<double> A(n * n), y(n), c(n);
    for (int i = 0; i < n; ++i) {
        const double t = std::cos(M_PI * (i + 0.5) / n);
        y[i] = f((t + 1) * 0.5 * zmax);
        double p = 1;
        for (int j = 0; j < n; ++j) { A[i * n + j] = p; p *= t; }
    }
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i) if (std::fabs(A[i * n + k]) > std::fabs(A[piv * n + k])) piv = i;
        for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]);
        std::swap(y[k], y[piv]);
        for (int i = k + 1; i < n; ++i) {
            const double r = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; ++j) A[i * n + j] -= r * A[k * n + j];
            y[i] -= r * y[k];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        double s = y[k];
        for (int j = k + 1; j < n; ++j) s -= A[k * n + j] * c[j];
        c[k] = s / A[k * n + k];
    }
    for (int k = 0; k < n; ++k) out[k] = (float)(scale * c[k]);
}

/* Beta and the coulomb constant of an Ewald real-space energy column e(k)
 * (row k at r2 = cd / (k + 1)): beta by bisection on the ratio of rows k1
 * (r = cutoff / sqrt(2)) and k2 (r = cutoff / 10), which falls monotonically
 * with beta; the constant from row k2.  False unless both are positive and
 * finite. */
template <class E>
inline bool ew_beta_coul(E eel, double cd, int k1, int k2, double *beta_out, double *coul_out)
{
    auto rr = [&](int k) { return std::sqrt(cd / (k + 1)); };
    const double target = eel(k1) / eel(k2);
    auto f = [&](double b) {
        return (std::erfc(b * rr(k1)) / rr(k1)) / (std::erfc(b * rr(k2)) / rr(k2)) - target;
    };
    double lo = 1e-3, hi = 10.0;
    if (!(f(lo) > 0 && f(hi) < 0)) return false;
    for (int it = 0; it < 200; ++it) {
        const double m = 0.5 * (lo + hi);
        if (f(m) > 0) lo = m; else hi = m;
    }
    const double beta = 0.5 * (lo + hi);
    const double coul = eel(k2) * rr(k2) / std::erfc(beta * rr(k2));
    if (!(beta > 0 && coul > 0 && std::isfinite(beta) && std::isfinite(coul))) return false;
    *beta_out = beta;
    *coul_out = coul;
    return true;
}

/* Correction polynomials for P's beta, tA and switch: the fewest terms whose
 * FP32 evaluation keeps the energy correction within 3e-7 of 1/r and the
 * force one within 1.5e-6 of 1/r^3 over [0, cutoff]; false if neither
 * count reaches it. */
inline bool ew_poly(double beta, double zmax, double coul, nbc_par *P, double *coul_out)
{
    for (int np : { kNpShort, kNpLong }) {
        chebfit(ew_g, zmax, np, beta * beta * beta, P->pf);
        chebfit(ew_h, zmax, np, 1.0, P->pe);
        double eg = 0, eh = 0;
        for (int i = 0; i <= 4096; ++i) {
            const double z = zmax * i / 4096.0;
            const float t = std::fmaf((float)z, P->tA / (float)(beta * beta), -1.f);
            float pg = P->pf[np - 1], ph = P->pe[np - 1];
            for (int k = np - 2; k >= 0; --k) { pg = std::fmaf(pg, t, P->pf[k]); ph = std::fmaf(ph, t, P->pe[k]); }
            eg = std::max(eg, std::fabs(pg / (beta * beta * beta) - ew_g(z)) * z * std::sqrt(z));
            eh = std::max(eh, std::fabs(ph - ew_h(z)) * std::sqrt(z));
        }
        bool finite = std::isfinite(P->beta) && std::isfinite(P->tA) && std::isfinite(P->sw_a) &&
                      std::isfinite(P->sw_b) && std::isfinite(P->sw_d) && std::isfinite(P->sw_e);
        for (int k = 0; k < np; ++k) finite = finite && std::isfinite(P->pf[k]) && std::isfinite(P->pe[k]);
        for (int k = 0; k < 8; ++k) finite = finite && std::isfinite(P->fs[k]);
        if (finite && eh <= 3e-7 && eg <= 1.5e-6) {
            P->np = np;
            *coul_out = coul;
            return true;
        }
    }
    return false;
}

/* Fits the linear PME table (row k at s = k + 1 = cutoff2*density/r2; per row
 * lj12, lj6, elec) as Ewald real space plus LJ with CHARMM's potential switch
 * or force switch (Steinbach and Brooks, J. Comput. Chem. 15, 667 (1994)):
 * succeeds when some beta, coulomb constant and switch radius reproduce every
 * row inside the cutoff to 1e-8, and returns the analytic parameters.
 * Otherwise false.
 * Rows 0 .. density-2 lie beyond the cutoff (zero), row density-1 is at the
 * cutoff, rows density .. nrow-2 are compared, and the terminal row nrow-1 is
 * never filled and not compared.  Any non-finite value declines the fit. */
inline bool nbc_fit_ewald(const double *tg, const double *te, int nrow, double density, double cutoff2,
                   nbc_par *P, double *coul_out)
{
    for (long long i = 0; i < 3LL * nrow; ++i)
        if (!std::isfinite(tg[i]) || !std::isfinite(te[i])) return false;
    const double cd = cutoff2 * density;
    const int k0 = (int)density, k1 = 2 * k0, k2 = 100 * k0;
    if (nrow < k2 + 2 || k0 < 2 || k1 <= k0) return false;
    double beta, coul;
    if (!ew_beta_coul([&](int k) { return te[3 * k + 2]; }, cd, k1, k2, &beta, &coul)) return false;
    /* the LJ gradient rows are pure r^-12 below the switch radius */
    auto pure12 = [&](int k) { const double r2 = cd / (k + 1), ri2 = 1 / r2, ri6 = ri2 * ri2 * ri2;
                               return -12 * ri6 * ri6 * ri2; };
    int kin = -1;   /* the largest-r row inside the switch */
    for (int k = k0; k < nrow - 1; ++k)
        if (std::fabs(tg[3 * k] - pure12(k)) > 1e-9 * std::fabs(pure12(k))) kin = k;
        else break;
    if (kin < 0) return false;
    const double c3 = cutoff2 * cutoff2 * cutoff2, c15 = cutoff2 * std::sqrt(cutoff2);
    auto lj_rows = [&](int fsw, int k, double ron2, double *g12, double *g6, double *e12, double *e6) {
        const double r2 = cd / (k + 1), ri2 = 1 / r2, ri6 = ri2 * ri2 * ri2;
        *g12 = -12 * ri6 * ri6 * ri2; *g6 = -6 * ri6 * ri2; *e12 = ri6 * ri6; *e6 = ri6;
        if (fsw) {
            const double o3 = ron2 * ron2 * ron2, o15 = ron2 * std::sqrt(ron2), ri3 = std::sqrt(ri6);
            if (r2 > ron2) {
                const double k12 = c3 / (c3 - o3), k6 = c15 / (c15 - o15);
                *g12 = -12 * k12 * (ri6 - 1 / c3) * ri6 * ri2; *g6 = -6 * k6 * (ri3 - 1 / c15) * ri3 * ri2;
                *e12 = k12 * (ri6 - 1 / c3) * (ri6 - 1 / c3); *e6 = k6 * (ri3 - 1 / c15) * (ri3 - 1 / c15);
            } else {
                *e12 -= 1 / (c3 * o3); *e6 -= 1 / (c15 * o15);
            }
        } else if (r2 > ron2 && r2 < cutoff2) {
            const double sw = 1 / std::pow(cutoff2 - ron2, 3), t1 = cutoff2 - r2;
            const double S = t1 * t1 * (cutoff2 + 2 * r2 - 3 * ron2) * sw, dS = 12 * t1 * (ron2 - r2) * sw;
            *g12 = S * *g12 + dS * *e12; *g6 = S * *g6 + dS * *e6; *e12 *= S; *e6 *= S;
        }
    };
    for (int fsw = 0; fsw < 2; ++fsw) {
        double ron2 = cutoff2;
        const double r2in = cd / (kin + 1), ri6in = 1 / (r2in * r2in * r2in);
        if (!fsw) {   /* solve that row's switch value for ron2 */
            const double S = te[3 * kin] / (ri6in * ri6in);
            double lo = 0.0, hi = r2in;
            for (int it = 0; it < 200; ++it) {
                /* the switch at r2in rises monotonically with ron2 to 1 at r2in */
                const double m = 0.5 * (lo + hi), t1 = cutoff2 - r2in;
                const double Sm = t1 * t1 * (cutoff2 + 2 * r2in - 3 * m) / std::pow(cutoff2 - m, 3);
                if (Sm > S) hi = m; else lo = m;
            }
            ron2 = 0.5 * (lo + hi);
        } else {   /* that row's gradient over the pure one is k12 (1 - r^6/roff^6) */
            const double k12 = (tg[3 * kin] / pure12(kin)) / (1 - r2in * r2in * r2in / c3);
            const double o3 = c3 * (1 - 1 / k12);
            if (!(k12 > 1 && std::isfinite(k12) && o3 > 0)) continue;
            ron2 = std::cbrt(o3);
        }
        if (!(ron2 > 0 && ron2 < r2in && cutoff2 - ron2 > 1e-6 * cutoff2)) continue;
        bool match = true;
        for (int k = k0; match && k < nrow - 1; ++k) {
            const double r2 = cd / (k + 1);
            if (r2 >= cutoff2) continue;
            double g12, g6, e12, e6;
            lj_rows(fsw, k, ron2, &g12, &g6, &e12, &e6);
            const double r = std::sqrt(r2), z = beta * beta * r2;
            const double gel = -coul * (1 / (r * r * r) - beta * beta * beta * ew_g(z));
            const double eel = coul * std::erfc(beta * r) / r;
            const double want[6] = { g12, g6, gel, e12, e6, eel };
            const double have[6] = { tg[3 * k], tg[3 * k + 1], tg[3 * k + 2], te[3 * k], te[3 * k + 1], te[3 * k + 2] };
            for (int e = 0; e < 6; ++e)
                if (!std::isfinite(want[e]) ||
                    std::fabs(have[e] - want[e]) > 1e-8 * std::fabs(want[e]) + 1e-300) match = false;
        }
        if (!match) continue;
        /* erfc(beta rc) between 1e-3 and 1e-30 */
        const double zmax = beta * beta * cutoff2;
        if (!(zmax >= 5.4 && zmax <= 64)) return false;
        const double D = std::pow(cutoff2 - ron2, 3);
        P->beta = (float)beta;
        P->tA = (float)(beta * beta * 2.0 / zmax);
        P->ron2 = (float)ron2;
        P->roff2 = (float)cutoff2;
        P->fsw = fsw;
        if (!fsw) {
            P->sw_a = (float)(2.0 / D);
            P->sw_b = (float)((cutoff2 - 3 * ron2) / D);
            P->sw_d = (float)(-12.0 / D);
            P->sw_e = (float)(12.0 * ron2 / D);
        } else {
            const double o3 = ron2 * ron2 * ron2, o15 = ron2 * std::sqrt(ron2);
            const double k12 = c3 / (c3 - o3), k6 = c15 / (c15 - o15);
            const double f[8] = { k12, k12 / c3, k12 / (c3 * c3), 1 / (c3 * o3),
                                  k6, k6 / c15, k6 / (c15 * c15), 1 / (c15 * o15) };
            P->sw_a = P->sw_b = P->sw_d = P->sw_e = 0.f;
            for (int i = 0; i < 8; ++i) P->fs[i] = (float)f[i];
        }
        return ew_poly(beta, zmax, coul, P, coul_out);
    }
    return false;
}

/* Fits the electrostatic-only table (GC_TABLE_PME_ELEC, vdw = CUTOFF; one
 * value per row) as Ewald real space to 1e-8 inside the cutoff.  The
 * analytic parameters carry no switch: ron2 = roff2 = cutoff2, so S = 1,
 * dS = 0 and the LJ term is the plain pair. */
inline bool nbc_fit_ewald_elec(const double *tg, const double *te, int nrow, double density,
                               double cutoff2, nbc_par *P, double *coul_out)
{
    for (int i = 0; i < nrow; ++i)
        if (!std::isfinite(tg[i]) || !std::isfinite(te[i])) return false;
    const double cd = cutoff2 * density;
    const int k0 = (int)density, k1 = 2 * k0, k2 = 100 * k0;
    if (nrow < k2 + 2 || k0 < 2 || k1 <= k0) return false;
    double beta, coul;
    if (!ew_beta_coul([&](int k) { return te[k]; }, cd, k1, k2, &beta, &coul)) return false;
    for (int k = k0; k < nrow - 1; ++k) {
        const double r2 = cd / (k + 1);
        if (r2 >= cutoff2) continue;
        const double r = std::sqrt(r2), z = beta * beta * r2;
        const double want[2] = { -coul * (1 / (r * r * r) - beta * beta * beta * ew_g(z)),
                                 coul * std::erfc(beta * r) / r };
        const double have[2] = { tg[k], te[k] };
        for (int e = 0; e < 2; ++e)
            if (!std::isfinite(want[e]) ||
                std::fabs(have[e] - want[e]) > 1e-8 * std::fabs(want[e]) + 1e-300) return false;
    }
    const double zmax = beta * beta * cutoff2;
    if (!(zmax >= 5.4 && zmax <= 64)) return false;
    P->beta = (float)beta;
    P->tA = (float)(beta * beta * 2.0 / zmax);
    P->ron2 = (float)cutoff2;
    P->roff2 = (float)cutoff2;
    P->sw_a = P->sw_b = P->sw_d = P->sw_e = 0.f;
    return ew_poly(beta, zmax, coul, P, coul_out);
}

}  /* namespace gcn */

#endif
