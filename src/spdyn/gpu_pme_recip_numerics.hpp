/* Per-atom PME numerics of the distributed reciprocal component:
 * bspline_dev, grid_locate, build_b2 and make_recip_params, the same
 * arithmetic as GENESIS's sp_energy_pme_opt_1dalltoall.fpp and gpu_pme.cu's
 * native_pme_plan (r_scale, gfact, bs_fact3, bs_fact3d, vol_fact2, vol_fact4,
 * theta_fact).  Only where the arithmetic writes differs (a brick and a Z
 * pencil rather than one whole mesh); that lives in gpu_pme_recip.cu.
 * pme_reference.hpp shares none of this code and is the independent check:
 * closed-form B-splines and Essmann's form of the Ewald constants.
 */
#ifndef GENESIS_NATIVE_S4PME_RECIP_NUMERICS_HPP
#define GENESIS_NATIVE_S4PME_RECIP_NUMERICS_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__CUDACC__)
#define S4PME_HD __host__ __device__ __forceinline__
#else
#define S4PME_HD inline
#endif

namespace genesis_native_s4pme {

/* GCN_PME_MAX_BS in Core's gpu_core_native.h. */
enum { kMaxOrder = 8 };

/* Field for field gpu_core_native.h's `struct gcn_pme_params`, so the two
 * structs copy member by member with no unit or convention change. */
struct RecipParams {
  int N[3];              /* mesh, x fastest in the real grid                 */
  int nbs;               /* B-spline order                                   */
  double r_scale[3];     /* N / box                                          */
  double gfact[3];       /* 2 pi / box                                       */
  double bs_fact3;       /* spread normalisation                             */
  double bs_fact3d;      /* gather normalisation, bs_fact3 * (n-1)           */
  double vol_fact2;      /* 2 pi el_fact / V, the energy factor              */
  double vol_fact4;      /* 2 * vol_fact2, the force factor                  */
  double theta_fact;     /* -1/(4 alpha^2)                                   */
};

/* What gc_pme_desc plus its box carry. */
struct RecipInput {
  double box[3];
  int ngrid[3];
  int order;
  double alpha;
  double elecoef;        /* GENESIS's ELECOEF                                */
  double dielec;
};

/* B-spline coefficients and derivatives.  Orders 4 and 6 are closed form;
 * anything else uses the recurrence.  T is double, or float for
 * nonbond_precision = MIXED (the offset u is exact in either).  M[j] is
 * (n-1)! times the cardinal B-spline at u + j; bs_fact3 and bs_fact3d undo
 * the factorials. */
template <typename T>
S4PME_HD void bspline_dev(int n, T u, T *M, T *dM)
{
  if (n == 4) {
    T u_3 = u * u * u;
    T u_2 = u * u;
    M[0] = u_3;
    M[1] = -T(3.0)*u_3 + T(3.0)*u_2 + T(3.0)*u + T(1.0);
    M[2] =  T(3.0)*u_3 - T(6.0)*u_2 + T(4.0);
    M[3] = (T(1.0) - u) * (T(1.0) - u) * (T(1.0) - u);
    dM[0] = u_2;
    dM[1] = -T(3.0)*u_2 + T(2.0)*u + T(1.0);
    dM[2] =  T(3.0)*u_2 - T(4.0)*u;
    dM[3] = -(T(1.0) - u) * (T(1.0) - u);
  } else if (n == 6) {
    T u_2 = u*u, u_3 = u_2*u, u_4 = u_3*u, u_5 = u_4*u;
    T v_1 = T(1.0) - u, v_2 = v_1*v_1, v_4 = v_2*v_2, v_5 = v_4*v_1;
    M[0] = u_5;
    M[1] = -T(5.0)*u_5 + T(5.0)*u_4 + T(10.0)*u_3 + T(10.0)*u_2 + T(5.0)*u + T(1.0);
    M[2] = T(10.0)*u_5 - T(20.0)*u_4 - T(20.0)*u_3 + T(20.0)*u_2 + T(50.0)*u + T(26.0);
    M[3] = -T(10.0)*u_5 + T(30.0)*u_4 - T(60.0)*u_2 + T(66.0);
    M[4] = T(5.0)*u_5 - T(20.0)*u_4 + T(20.0)*u_3 + T(20.0)*u_2 - T(50.0)*u + T(26.0);
    M[5] = v_5;
    dM[0] = u_4;
    dM[1] = -T(5.0)*u_4 + T(4.0)*u_3 + T(6.0)*u_2 + T(4.0)*u + T(1.0);
    dM[2] = T(10.0)*u_4 - T(16.0)*u_3 - T(12.0)*u_2 + T(8.0)*u + T(10.0);
    dM[3] = -T(10.0)*u_4 + T(24.0)*u_3 - T(24.0)*u;
    dM[4] = T(5.0)*u_4 - T(16.0)*u_3 + T(12.0)*u_2 + T(8.0)*u - T(10.0);
    dM[5] = -v_4;
  } else {
    int i, j;
    M[0] = u;
    M[1] = T(1.0) - u;
    for (j = 3; j <= n - 1; ++j) {
      M[j-1] = M[j-2] * (T(1.0) - u);
      for (i = 1; i <= j - 2; ++i)
        M[j-i-1] = ((T)(j-i-1) + u) * M[j-i-1]
                 + ((T)(i+1) - u) * M[j-i-2];
      M[0] = M[0] * u;
    }
    dM[0] = M[0];
    for (i = 2; i <= n - 1; ++i) dM[i-1] = M[i-1] - M[i-2];
    dM[n-1] = -M[n-2];
    M[n-1] = M[n-2] * (T(1.0) - u);
    for (i = 1; i <= n - 2; ++i)
      M[n-i-1] = ((T)(n-i-1) + u) * M[n-i-1]
               + ((T)(i+1) - u) * M[n-i-2];
    M[0] = M[0] * u;
  }
}

/* Coordinate to grid index and fractional offset: the wrap, the half-grid
 * shift and the 1e-7 clamp are GENESIS's own. */
S4PME_HD void grid_locate(double x, double r_scale, int N, int *ii, double *dv)
{
  double grid = (double)N;
  double half = (double)(N / 2);
  double vr = x * r_scale;
  vr = vr + half - grid * round(vr / grid);
  vr = vr < grid - 1.0e-7 ? vr : grid - 1.0e-7;
  int i = (int)vr;
  if (i > N - 1) i = N - 1;
  *ii = i;
  *dv = vr - (double)i;
}

/* One axis's grid_locate constants, taken once per block on the device. */
struct GridAxis { double grid, half, inv; };
S4PME_HD GridAxis grid_axis(int N)
{
  GridAxis g;
  g.grid = (double)N;
  g.half = (double)(N / 2);
  g.inv = 1.0 / g.grid;
  return g;
}

/* grid_locate without its division, mostly: round(vr / grid) is the
 * integer nearest vr * (1/grid) unless that product lies near a half-integer
 * or is huge, where the division decides. */
S4PME_HD void grid_locate(double x, double r_scale, const GridAxis &g, int N,
                          int *ii, double *dv)
{
  double vr = x * r_scale;
  const double q = vr * g.inv, k = round(q);
  const double w = fabs(q) < 1.0e9 && fabs(q - k) < 0.499 ? k : round(vr / g.grid);
  vr = vr + g.half - g.grid * w;
  vr = vr < g.grid - 1.0e-7 ? vr : g.grid - 1.0e-7;
  int i = (int)vr;
  if (i > N - 1) i = N - 1;
  *ii = i;
  *dv = vr - (double)i;
}

/* grid_locate in FP32 (nonbond_precision = MIXED): the coordinate is scaled
 * and wrapped in float and the clamp is the largest float below the grid.
 * Spread and gather call the same function, so an atom's stencil is the same
 * cells in both. */
struct GridAxisF { float r_scale, grid, half, inv, top; };
S4PME_HD GridAxisF grid_axis_f(int N, double r_scale)
{
  GridAxisF g;
  g.r_scale = (float)r_scale;
  g.grid = (float)N;
  g.half = (float)(N / 2);
  g.inv = 1.0f / g.grid;
  g.top = nextafterf(g.grid, 0.0f);
  return g;
}
S4PME_HD void grid_locate(double x, const GridAxisF &g, int N, int *ii, float *dv)
{
  float vr = (float)x * g.r_scale;
  vr = vr + g.half - g.grid * roundf(vr * g.inv);
  vr = vr < g.top ? vr : g.top;
  int i = (int)vr;
  if (i > N - 1) i = N - 1;
  *ii = i;
  *dv = vr - (float)i;
}

/* The Euler exponential spline factor b2(h) = 1/|b(h)|^2: the B-spline
 * coefficients at u = 1, scaled by bs_fact, then the squared modulus of
 * their discrete Fourier sum. */
inline void build_b2(int n_bs, int N, double bs_fact, std::vector<double> &out)
{
  double M[kMaxOrder], dM[kMaxOrder];
  bspline_dev(n_bs, 1.0, M, dM);

  std::vector<double> bs((std::size_t)n_bs);
  for (int i = 0; i < n_bs; ++i) bs[(std::size_t)i] = M[i];
  for (int i = 0; i < n_bs - 1; ++i) bs[(std::size_t)i] *= bs_fact;

  const double two_pi = 6.28318530717958647692;
  const double fact = two_pi / (double)N;

  out.resize((std::size_t)N);
  for (int j = 0; j < N; ++j) {
    int js = (j <= N / 2) ? j : j - N;
    double bcos = 0.0, bsin = 0.0;
    for (int i = 0; i <= n_bs - 2; ++i) {
      bcos += bs[(std::size_t)i] * std::cos((double)(js * i) * fact);
      bsin += bs[(std::size_t)i] * std::sin((double)(js * i) * fact);
    }
    out[(std::size_t)j] = 1.0 / (bcos * bcos + bsin * bsin);
  }
}

/* bs_fact = 1 / prod_{i=1}^{n-2} (n-i), as native_pme_plan forms it. */
inline double bs_fact_of(int order)
{
  double j = 1.0;
  for (int i = 1; i <= order - 2; ++i) j *= (double)(order - i);
  return 1.0 / j;
}

/* native_pme_plan's arithmetic.  Refuses what would reach the kernels as
 * an infinity or an out-of-range order; the limits that bind only spread
 * and gather are in spread_gather_admissible. */
inline RecipParams make_recip_params(const RecipInput &in)
{
  if (in.order < 2 || in.order > kMaxOrder)
    throw std::invalid_argument("s4pme: the B-spline order is outside [2,8]");
  for (int k = 0; k < 3; ++k) {
    if (in.ngrid[k] <= 0)
      throw std::invalid_argument("s4pme: a mesh axis is not positive");
    if (!(in.box[k] > 0.0) || !std::isfinite(in.box[k]))
      throw std::invalid_argument("s4pme: a box length is not positive");
  }
  if (!(in.alpha > 0.0) || !std::isfinite(in.alpha))
    throw std::invalid_argument("s4pme: alpha is not positive");
  if (!(in.dielec > 0.0) || !std::isfinite(in.elecoef))
    throw std::invalid_argument("s4pme: the dielectric is not positive");

  const double bs_fact = bs_fact_of(in.order);
  const double bs_fact3 = bs_fact * bs_fact * bs_fact;
  const double pi = 3.14159265358979323846;
  const double el_fact = in.elecoef / in.dielec;

  RecipParams p;
  double inv_volume = 1.0;
  for (int k = 0; k < 3; ++k) {
    p.N[k] = in.ngrid[k];
    p.r_scale[k] = (double)in.ngrid[k] / in.box[k];
    p.gfact[k] = 2.0 * pi / in.box[k];
    inv_volume /= in.box[k];
  }
  p.nbs = in.order;
  p.bs_fact3 = bs_fact3;
  p.bs_fact3d = bs_fact3 * (double)(in.order - 1);
  p.vol_fact2 = 2.0 * pi * el_fact * inv_volume;
  p.vol_fact4 = 2.0 * p.vol_fact2;
  p.theta_fact = -0.25 / (in.alpha * in.alpha);
  return p;
}

/* The two limits spread and gather need.  Returns 0 when admissible, else
 * the reason.
 *
 *   - an axis shorter than the order: the stencil wraps a cell index by
 *     adding N once, which is only enough while N >= order - 1.
 *   - an ODD axis: for odd N, grid_locate can give vr in [-1/2, 0) for an
 *     atom at the half-box plane; int() truncates that to cell 0 with a
 *     NEGATIVE offset u, and bspline_dev evaluates outside [0,1).  GENESIS
 *     forces ngrid_x even but not y or z, so this is reachable. */
inline const char *spread_gather_admissible(const RecipParams &p)
{
  for (int k = 0; k < 3; ++k) {
    if (p.N[k] < p.nbs)
      return "a mesh axis is shorter than the B-spline order; the stencil "
             "would wrap more than once";
    if (p.N[k] % 2 != 0)
      return "a mesh axis is odd; GENESIS's grid_locate gives a negative "
             "spline offset in a half-cell sliver on an odd axis";
  }
  return 0;
}

} /* namespace genesis_native_s4pme */

#endif
