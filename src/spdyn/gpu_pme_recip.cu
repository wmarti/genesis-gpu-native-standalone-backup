/* Distributed PME reciprocal component, device half.
 *
 * The kernels are gcn_kern_pme_spread / _solve / _gather (gpu_pme.cu) with the
 * changes gpu_pme_recip.hpp states: spread and gather address one brick of the
 * mesh, and the solve addresses one Z pencil and reduces all six virial
 * components deterministically instead of by atomics.  Formulas, factors,
 * cut-offs and association inside the loops are unchanged.
 */
#include "gpu_pme_recip.hpp"
#include "gpu_fixed_sum.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#ifdef S4PME_SYNTAX_CHECK
#define S4PME_LAUNCH(fn, grid, blk, stm) fn
#else
#define S4PME_LAUNCH(fn, grid, blk, stm) fn<<<(grid), (blk), 0, (stm)>>>
#endif

namespace genesis_native_s4pme {

namespace {

const int kBlock = 128;          /* GCN_PME_BLOCK                           */
const int kMaxSolveBlocks = 1024;

/* The brick's sums per precision: FP64 terms in 64-bit words (2^-44),
 * FP32 terms (MIXED) in 32-bit words (2^-24), gpu_fixed_sum.cuh. */
template <typename T> struct BrickSum;
template <> struct BrickSum<double> {
  typedef gcn_fx::word W;
  enum { R = gcn_fx::kCharge };
};
template <> struct BrickSum<float> {
  typedef unsigned int W;
  enum { R = gcn_fx::kCharge32 };
};

/* The spread is lane-mapped: nbs threads per atom, one per stencil column
 * along x (the brick's contiguous axis), so an atom's lanes add to adjacent
 * cells of one row.  An atom's B-splines are computed once, one lane per
 * axis, and shared through shared memory; each lane then walks its nbs*nbs
 * (y, z) cells.  At most kLaneAtoms atoms per block of at most 256 threads. */
const int kLaneAtoms = 64;

__host__ __device__ inline int lane_atoms(int nbs)
{
  return 256 / nbs < kLaneAtoms ? 256 / nbs : kLaneAtoms;
}

/* This block's atoms' grid points and B-splines into shared memory: lane
 * d < 3 of an atom takes axis d (lanes 0 and 1 take all three between them
 * when nbs == 2). */
template <typename T>
__device__ __forceinline__ void lane_splines(
    const double *__restrict__ coord, long pitch, const RecipParams &prm,
    long a, int slot, int lane, T (*th)[kMaxOrder][kLaneAtoms],
    int (*gi)[kLaneAtoms])
{
  const int nbs = prm.nbs;
  for (int d = lane; d < 3; d += nbs) {
    int ii;
    double dv;
    grid_locate(coord[d * pitch + a], prm.r_scale[d], prm.N[d], &ii, &dv);
    T M[kMaxOrder], dM[kMaxOrder];
    bspline_dev(nbs, (T)dv, M, dM);
    gi[d][slot] = ii;
    for (int j = 0; j < nbs; ++j) th[d][j][slot] = M[j];
  }
}

/* A mesh cell's place in the brick along one axis, or -1 outside it: the
 * brick is cells b0, b0+1, ... b0+n-1 of the axis modulo N (BrickBox). */
__device__ __forceinline__ long brick_at(int g, long b0, long n, int N)
{
  long l = (long)g - b0;
  if (l < 0) l += N;
  return l < n ? l : -1;
}

/* A stencil cell outside the brick poisons the unit's sums, so every fold
 * reads NaN (gpu_pme_recip.hpp, spread). */
__device__ __forceinline__ void brick_missed() { atomicOr(&gcn_fx::overflow, 1u); }

/* One lane's nbs*nbs stencil cells of the atom in `slot`, added to the
 * brick's order-independent fixed-point sums (BrickSum); k_fold_brick turns
 * those into a mesh.  A stencil cell is wrapped onto the global mesh first,
 * then placed in the brick. */
template <typename T>
__device__ __forceinline__ void lane_adds(
    const RecipParams &prm, const BrickBox &b,
    typename BrickSum<T>::W *__restrict__ fx, int slot, int lane,
    const T (*th)[kMaxOrder][kLaneAtoms], const int (*gi)[kLaneAtoms],
    const T *qs)
{
  const int nbs = prm.nbs;
  int gx = gi[0][slot] - lane; if (gx < 0) gx += prm.N[0];
  const long lx = brick_at(gx, b.x0, b.nx, prm.N[0]);
  if (lx < 0) { brick_missed(); return; }
  const T mx = th[0][lane][slot], qtmp = qs[slot];
  for (int ky = 0; ky < nbs; ++ky) {
    int gy = gi[1][slot] - ky; if (gy < 0) gy += prm.N[1];
    const long ly = brick_at(gy, b.y0, b.ny, prm.N[1]);
    if (ly < 0) { brick_missed(); return; }
    const T mxy = mx * th[1][ky][slot];
    for (int kz = 0; kz < nbs; ++kz) {
      int gz = gi[2][slot] - kz; if (gz < 0) gz += prm.N[2];
      const long lz = brick_at(gz, b.z0, b.nz, prm.N[2]);
      if (lz < 0) { brick_missed(); return; }
      gcn_fx::add1<BrickSum<T>::R>(fx + (lz * b.ny + ly) * b.nx + lx,
                                   mxy * th[2][kz][slot] * qtmp);
    }
  }
}

/* Charge spread into one brick, deterministically (lane_adds). */
template <typename T>
__global__ void __launch_bounds__(256)
k_spread_brick_lanes(const double *__restrict__ coord,
                     const double *__restrict__ charge,
                     long natoms, long pitch, RecipParams prm,
                     BrickBox b, typename BrickSum<T>::W *__restrict__ fx)
{
  __shared__ T th[3][kMaxOrder][kLaneAtoms];
  __shared__ int gi[3][kLaneAtoms];
  __shared__ T qs[kLaneAtoms];
  const int nbs = prm.nbs, per = lane_atoms(nbs);
  const int slot = threadIdx.x / nbs, lane = threadIdx.x - slot * nbs;
  const long a = (long)blockIdx.x * per + slot;
  const bool live = slot < per && a < natoms;
  if (live) {
    lane_splines<T>(coord, pitch, prm, a, slot, lane, th, gi);
    if (lane == 0) qs[slot] = (T)(charge[a] * prm.bs_fact3);
  }
  __syncthreads();
  if (live) lane_adds<T>(prm, b, fx, slot, lane, th, gi, qs);
}

/* The MIXED spread with a shared-memory pre-sum.  Slots are grouped by cell,
 * so kTileAtoms consecutive atoms cover a small box of mesh points; their
 * 32-bit words are summed there first and each touched point costs one
 * global add.  Sums modulo 2^32 are the same bits as add1 adding directly.
 * A block whose box exceeds kTileCells is spread as its two halves, each in
 * its own box; a half still exceeding it adds directly (lane_adds). */
const int kTileAtoms = 256;
const int kTileHalf = kTileAtoms / 2;
const int kTileCells = 6144;     /* 24 KB of words */

template <int NBS>
__global__ void __launch_bounds__(256)
k_spread_brick_tile(const double *__restrict__ coord,
                    const double *__restrict__ charge,
                    long natoms, long pitch, RecipParams prm,
                    BrickBox b, unsigned int *__restrict__ fx)
{
  __shared__ float th[3][kMaxOrder][kLaneAtoms];
  __shared__ int gi[3][kLaneAtoms];
  __shared__ float qs[kLaneAtoms];
  __shared__ unsigned int tile[kTileCells];
  __shared__ int lo[3][3], hi[3][3];     /* [the block, its halves][axis]   */
  __shared__ GridAxisF ax[3];
  __shared__ float bs3;
  __shared__ int ga[3][kTileAtoms];      /* the atoms' grid points and      */
  __shared__ float gv[3][kTileAtoms];    /* offsets, for their B-splines    */
  const int nbs = NBS ? NBS : prm.nbs, per = lane_atoms(nbs), t = threadIdx.x;
  const long a0 = (long)blockIdx.x * kTileAtoms;
  const int na = natoms - a0 < kTileAtoms ? (int)(natoms - a0) : kTileAtoms;

  /* The halves' boxes: grid points lo-nbs+1 .. hi per axis, unwrapped; the
     block's is their union (an empty half's box is inverted). */
  if (t < 3) {
    for (int k = 1; k < 3; ++k) { lo[k][t] = prm.N[t]; hi[k][t] = -1; }
    ax[t] = grid_axis_f(prm.N[t], prm.r_scale[t]);
  }
  if (t == 0) bs3 = (float)prm.bs_fact3;
  __syncthreads();
  for (int s = t; s < na; s += blockDim.x) {
    const int h = s < kTileHalf ? 1 : 2;
    for (int d = 0; d < 3; ++d) {
      int ii;
      float dv;
      grid_locate(coord[d * pitch + a0 + s], ax[d], prm.N[d], &ii, &dv);
      ga[d][s] = ii;
      gv[d][s] = dv;
      atomicMin(&lo[h][d], ii);
      atomicMax(&hi[h][d], ii);
    }
  }
  __syncthreads();
  if (t < 3) { lo[0][t] = min(lo[1][t], lo[2][t]); hi[0][t] = max(hi[1][t], hi[2][t]); }
  __syncthreads();
  auto box_cells = [&](int k) {
    return (hi[k][0] - lo[k][0] + nbs) * (hi[k][1] - lo[k][1] + nbs) * (hi[k][2] - lo[k][2] + nbs);
  };
  const bool whole = box_cells(0) <= kTileCells;

  const int slot = t / nbs, lane = t - slot * nbs;
  for (int k = whole ? 0 : 1; k < (whole ? 1 : 3); ++k) {
    const int c0 = k == 2 ? kTileHalf : 0, c1 = k == 1 ? min(na, kTileHalf) : na;
    if (c1 <= c0) break;
    const int ex = hi[k][0] - lo[k][0] + nbs, ey = hi[k][1] - lo[k][1] + nbs;
    const int x0 = lo[k][0] - nbs + 1, y0 = lo[k][1] - nbs + 1, z0 = lo[k][2] - nbs + 1;
    const int cells = box_cells(k);
    const bool tiled = cells <= kTileCells;
    __syncthreads();                       /* the previous half's flush read */
    if (tiled)
      for (int s = t; s < cells; s += blockDim.x) tile[s] = 0;

    for (int c = c0; c < c1; c += per) {
      const bool live = slot < per && c + slot < c1;
      __syncthreads();                     /* previous chunk's splines read */
      if (live) {
        const int s = c + slot;
        for (int d = lane; d < 3; d += nbs) {
          float M[kMaxOrder], dM[kMaxOrder];
          bspline_dev(nbs, gv[d][s], M, dM);
          gi[d][slot] = ga[d][s];
          for (int j = 0; j < nbs; ++j) th[d][j][slot] = M[j];
        }
        if (lane == 0) qs[slot] = (float)charge[a0 + s] * bs3;
      }
      __syncthreads();
      if (!live) continue;
      if (!tiled) {
        lane_adds<float>(prm, b, fx, slot, lane, th, gi, qs);
        continue;
      }
      const int tx = gi[0][slot] - lane - x0;
      const float mx = th[0][lane][slot], qtmp = qs[slot];
      for (int ky = 0; ky < nbs; ++ky) {
        const float mxy = mx * th[1][ky][slot];
        const int row = (gi[1][slot] - ky - y0) * ex + tx;
        for (int kz = 0; kz < nbs; ++kz)
          gcn_fx::add1<BrickSum<float>::R>(
              tile + (gi[2][slot] - kz - z0) * ex * ey + row,
              mxy * th[2][kz][slot] * qtmp);
      }
    }
    if (!tiled) continue;
    __syncthreads();
    for (int s = t; s < cells; s += blockDim.x) {
      const unsigned int w = tile[s];
      if (w == 0) continue;
      const int iz = s / (ex * ey), iy = (s - iz * ex * ey) / ex;
      int gx = x0 + s - (iz * ey + iy) * ex; if (gx < 0) gx += prm.N[0];
      int gy = y0 + iy; if (gy < 0) gy += prm.N[1];
      int gz = z0 + iz; if (gz < 0) gz += prm.N[2];
      const long lx = brick_at(gx, b.x0, b.nx, prm.N[0]);
      const long ly = brick_at(gy, b.y0, b.ny, prm.N[1]);
      const long lz = brick_at(gz, b.z0, b.nz, prm.N[2]);
      if (lx < 0 || ly < 0 || lz < 0) { brick_missed(); continue; }
      atomicAdd(fx + (lz * b.ny + ly) * b.nx + lx, w);
    }
  }
}

/* A mesh from its sums, zeroing the sums for the next spread in the same
 * pass.  The FP32 mesh's words convert in FP32 (value1f). */
template <int R>
__device__ __forceinline__ float fold_value(unsigned int w) { return gcn_fx::value1f<R>(w); }
template <int R>
__device__ __forceinline__ double fold_value(gcn_fx::word w) { return gcn_fx::value1<R>(w); }

template <typename T, typename W, int R>
__global__ void k_fold_brick(W *__restrict__ fx, long n, T *__restrict__ brick,
                             const FoldSelf self)
{
  const W *__restrict__ in = (const W *)self.src;
  for (long s = (long)blockIdx.x * blockDim.x + threadIdx.x; s < n;
       s += (long)gridDim.x * blockDim.x) {
    W w = fx[s];
    if (self.n > 0) {
      /* 32-bit: the pencil's cells fit (fold() checks); 64-bit division is slow */
      const unsigned u = (unsigned)s, d0 = (unsigned)self.d0, d1 = (unsigned)self.d1;
      const unsigned z = u / d0, r = u - z * d0;
      const unsigned y = r / d1, x = r - y * d1;
      for (int b = 0; b < self.n; ++b) {
        const FoldSelf::Box &q = self.box[b];
        const unsigned dz = z - (unsigned)q.z0, dy = y - (unsigned)q.y0,
                       dx = x - (unsigned)q.x0;
        if (dz < (unsigned)q.nz && dy < (unsigned)q.ny && dx < (unsigned)q.nx)
          w += in[q.off + (long)dz * q.s0 + (long)dy * q.s1 + (long)dx * q.s2];
      }
    }
    brick[s] = (T)fold_value<R>(w);
    fx[s] = 0;
  }
}

/* The spectral pass on one Z pencil: theta, vir_fact, cut-offs and
 * half-spectrum weight as in gpu_pme.cu; the virial off-diagonals are the
 * same tqq * vir_fact * g_a * g_b the diagonal carries with the delta.
 * theta depends only on the box, alpha and the pencil, so it is tabulated:
 * Th = kThetaStore computes and stores it in `tab`, Th = kThetaLoad reads it
 * back, and a step without scalars is one multiply per point.  The FP32 mesh
 * (C = cufftComplex) also keeps theta in FP32 in `tabf` for steps without
 * scalars; steps with scalars use the FP64 table. */
enum { kThetaStore, kThetaLoad };

template <typename C, bool S, int Th>
__global__ void k_solve_zpencil(C *__restrict__ sp,
                                ZPencilBox zp, RecipParams prm,
                                const double *__restrict__ b2x,
                                const double *__restrict__ b2y,
                                const double *__restrict__ b2z,
                                double *__restrict__ tab,
                                float *__restrict__ tabf,
                                double *__restrict__ part)
{
  constexpr bool F32 = std::is_same<C, cufftComplex>::value;
  __shared__ double sh[kScalars][kBlock];
  const int tid = threadIdx.x;
  double acc[kScalars] = {0, 0, 0, 0, 0, 0, 0};
  const long npoint = zp.nky * zp.nkx * zp.nz;
  const int Nx = prm.N[0], Ny = prm.N[1], Nz = prm.N[2];

  for (long idx = (long)blockIdx.x * blockDim.x + tid; idx < npoint;
       idx += (long)gridDim.x * blockDim.x) {
    if (Th == kThetaLoad && !S && F32) {
      C s = sp[idx];
      const float theta = tabf[idx];
      s.x *= theta;
      s.y *= theta;
      sp[idx] = s;
      continue;
    }
    if (Th == kThetaLoad && !S) {
      C s = sp[idx];
      const double theta = tab[idx];
      s.x = (double)s.x * theta;
      s.y = (double)s.y * theta;
      sp[idx] = s;
      continue;
    }
    long ix, iy, iz;
    if (zp.kx_fastest) {
      ix = idx % zp.nkx;
      const long r = idx / zp.nkx;
      iy = r % zp.nky;
      iz = r / zp.nky;
    } else {
      iz = idx % zp.nz;
      const long r = idx / zp.nz;
      ix = r % zp.nkx;
      iy = r / zp.nkx;
    }
    const int kx = (int)(zp.kx0 + ix);
    const int ky = (int)(zp.ky0 + iy);
    const int kz = (int)iz;

    int is = kx;
    int js = (ky <= Ny / 2) ? ky : ky - Ny;
    int ks = (kz <= Nz / 2) ? kz : kz - Nz;

    double gx = prm.gfact[0] * (double)is;
    double gy = prm.gfact[1] * (double)js;
    double gz = prm.gfact[2] * (double)ks;
    double g2 = gx*gx + gy*gy + gz*gz;

    double theta, vir_fact;
    if (g2 > 1.0e-10) {
      vir_fact = -2.0 * (1.0 - g2 * prm.theta_fact) / g2;
      if (Th == kThetaLoad) {
        theta = tab[idx];
      } else {
        if (g2 * prm.theta_fact < -80.0) {
          theta = 0.0;
        } else {
          theta = b2x[kx] * b2y[ky] * b2z[kz]
                * exp(g2 * prm.theta_fact) / g2;
        }
        if (fabs(theta) < 1.0e-15) theta = 0.0;
      }
    } else {
      vir_fact = 0.0;
      theta    = 0.0;
    }
    if (Th == kThetaStore) {
      tab[idx] = theta;
      if (F32) tabf[idx] = (float)theta;
    }

    C s = sp[idx];
    const double sx = s.x, sy = s.y;
    double w = (kx == 0 || 2 * kx == Nx) ? 1.0 : 2.0;
    if (S) {
      double tqq = (sx*sx + sy*sy) * theta * prm.vol_fact2 * w;
      acc[0] += tqq;
      acc[1] += tqq * (1.0 + vir_fact * gx * gx);
      acc[2] += tqq * (      vir_fact * gx * gy);
      acc[3] += tqq * (      vir_fact * gx * gz);
      acc[4] += tqq * (1.0 + vir_fact * gy * gy);
      acc[5] += tqq * (      vir_fact * gy * gz);
      acc[6] += tqq * (1.0 + vir_fact * gz * gz);
    }

    s.x = sx * theta;
    s.y = sy * theta;
    sp[idx] = s;
  }

  if (!S) return;
  for (int i = 0; i < kScalars; ++i) sh[i][tid] = acc[i];
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off)
      for (int i = 0; i < kScalars; ++i) sh[i][tid] += sh[i][tid + off];
    __syncthreads();
  }
  if (tid == 0)
    for (int i = 0; i < kScalars; ++i) part[blockIdx.x * kScalars + i] = sh[i][0];
}

/* Deterministic finish over the block partials: one warp per scalar, lane
 * l adding blocks l, l+32, ... in order, then a fixed shuffle tree, so the
 * same pencil gives the same seven bits every run. */
__global__ void k_sum_scalars(const double *__restrict__ part, int nblk,
                              double *__restrict__ out)
{
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (w >= kScalars) return;
  double s = 0.0;
  for (int b = l; b < nblk; b += 32) s += part[b * kScalars + w];
  for (int off = 16; off > 0; off >>= 1)
    s += __shfl_down_sync(0xffffffffu, s, off);
  if (l == 0) out[w] = s;
}

/* A gather block's per-axis constants.  FP64 uses grid_locate and the force
 * factors of gpu_pme.cu; MIXED locates in FP32 (GridAxisF) and scales its FP32
 * stencil sums by one FP32 factor per axis. */
template <typename T> struct GatherAxes;
template <> struct GatherAxes<double> {
  GridAxis g[3];
  __device__ void init(int d, const RecipParams &prm) { g[d] = grid_axis(prm.N[d]); }
};
template <> struct GatherAxes<float> {
  GridAxisF g[3];
  float fs[3];
  __device__ void init(int d, const RecipParams &prm)
  {
    g[d] = grid_axis_f(prm.N[d], prm.r_scale[d]);
    fs[d] = (float)(prm.vol_fact4 * prm.bs_fact3d * prm.r_scale[d]);
  }
};

__device__ __forceinline__ void locate(double x, const RecipParams &prm,
                                       const GatherAxes<double> &ax, int d,
                                       int *ii, double *dv)
{
  grid_locate(x, prm.r_scale[d], ax.g[d], prm.N[d], ii, dv);
}
__device__ __forceinline__ void locate(double x, const RecipParams &prm,
                                       const GatherAxes<float> &ax, int d,
                                       int *ii, float *dv)
{
  grid_locate(x, ax.g[d], prm.N[d], ii, dv);
}

__device__ __forceinline__ void store_force(double q, const RecipParams &prm,
                                            const GatherAxes<double> &,
                                            double f0, double f1, double f2,
                                            double *force, long pitch)
{
  double qtmp = q * prm.vol_fact4 * prm.bs_fact3d;
  force[0]         = -(f0 * qtmp * prm.r_scale[0]);
  force[pitch]     = -(f1 * qtmp * prm.r_scale[1]);
  force[2 * pitch] = -(f2 * qtmp * prm.r_scale[2]);
}
__device__ __forceinline__ void store_force(double q, const RecipParams &,
                                            const GatherAxes<float> &ax,
                                            float f0, float f1, float f2,
                                            double *force, long pitch)
{
  const float qf = (float)q;
  force[0]         = -(double)(f0 * (qf * ax.fs[0]));
  force[pitch]     = -(double)(f1 * (qf * ax.fs[1]));
  force[2 * pitch] = -(double)(f2 * (qf * ax.fs[2]));
}

/* The full-brick specialization keeps the stencil and arithmetic order, but
 * its wrapped grid indices are already local indices: no brick_at checks. */
template <typename T, bool Whole>
__device__ __forceinline__ void gather_stencil(const int ii[3],
                                                const T *Mx, const T *My,
                                                const T *Mz, const T *dMx,
                                                const T *dMy, const T *dMz,
                                                int nbs, int Nx, int Ny, int Nz,
                                                BrickBox b, const T *brick,
                                                T &f0, T &f1, T &f2)
{
  for (int jz = 0; jz < nbs; ++jz) {
    int gz = ii[2] - jz; if (gz < 0) gz += Nz;
    const long lz = Whole ? (long)gz : brick_at(gz, b.z0, b.nz, Nz);
    if (!Whole && lz < 0) continue;
    for (int jy = 0; jy < nbs; ++jy) {
      int gy = ii[1] - jy; if (gy < 0) gy += Ny;
      const long ly = Whole ? (long)gy : brick_at(gy, b.y0, b.ny, Ny);
      if (!Whole && ly < 0) continue;
      const T *row = brick + (lz * b.ny + ly) * b.nx;
      for (int jx = 0; jx < nbs; ++jx) {
        int gx = ii[0] - jx; if (gx < 0) gx += Nx;
        const long lx = Whole ? (long)gx : brick_at(gx, b.x0, b.nx, Nx);
        if (!Whole && lx < 0) continue;
        const T X = row[lx];
        f0 += dMx[jx] * My[jy]  * Mz[jz]  * X;
        f1 += Mx[jx]  * dMy[jy] * Mz[jz]  * X;
        f2 += Mx[jx]  * My[jy]  * dMz[jz] * X;
      }
    }
  }
}

/* One atom's force from one brick.  A cell outside the brick is skipped (the
 * spread of the same coordinates has already poisoned the mesh for it).
 * NBS > 0 is the order at compile time, so the B-splines stay in registers. */
template <typename T, int NBS>
__device__ __forceinline__ void gather_atom(const double *__restrict__ coord,
                                            const double *__restrict__ charge,
                                            long a, long pitch, const RecipParams &prm,
                                            const GatherAxes<T> &ax, const BrickBox &b,
                                            const T *__restrict__ brick,
                                            double *__restrict__ force)
{
  int ii[3];
  T   dv[3];
  locate(coord[a],             prm, ax, 0, &ii[0], &dv[0]);
  locate(coord[pitch + a],     prm, ax, 1, &ii[1], &dv[1]);
  locate(coord[2 * pitch + a], prm, ax, 2, &ii[2], &dv[2]);

  T Mx[kMaxOrder], My[kMaxOrder], Mz[kMaxOrder];
  T dMx[kMaxOrder], dMy[kMaxOrder], dMz[kMaxOrder];
  const int nbs = NBS ? NBS : prm.nbs;
  bspline_dev(nbs, dv[0], Mx, dMx);
  bspline_dev(nbs, dv[1], My, dMy);
  bspline_dev(nbs, dv[2], Mz, dMz);

  const int Nx = prm.N[0], Ny = prm.N[1], Nz = prm.N[2];
  T f0 = 0, f1 = 0, f2 = 0;

  const bool whole = b.x0 == 0 && b.y0 == 0 && b.z0 == 0 &&
                     b.nx == Nx && b.ny == Ny && b.nz == Nz;
  if (whole)
    gather_stencil<T, true>(ii, Mx, My, Mz, dMx, dMy, dMz,
                            nbs, Nx, Ny, Nz, b, brick, f0, f1, f2);
  else
    gather_stencil<T, false>(ii, Mx, My, Mz, dMx, dMy, dMz,
                             nbs, Nx, Ny, Nz, b, brick, f0, f1, f2);

  store_force(charge[a], prm, ax, f0, f1, f2, force + a, pitch);
}

/* Force gather from one brick, one atom per thread; NBS as in gather_atom,
 * one kernel per order so each keeps its own register budget. */
template <typename T, int NBS>
__global__ void k_gather_brick(const double *__restrict__ coord,
                               const double *__restrict__ charge,
                               long natoms, long pitch, RecipParams prm,
                               BrickBox b, const T *__restrict__ brick,
                               double *__restrict__ force)
{
  __shared__ GatherAxes<T> ax;
  if (threadIdx.x < 3) ax.init(threadIdx.x, prm);
  __syncthreads();
  long a = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= natoms) return;
  gather_atom<T, NBS>(coord, charge, a, pitch, prm, ax, b, brick, force);
}

int blocks_for(long n, int blk, int cap)
{
  long b = (n + blk - 1) / blk;
  if (b < 1) b = 1;
  if (b > cap) b = cap;
  return (int)b;
}

} /* namespace */

struct RecipDevice::Impl {
  RecipParams prm;
  cudaStream_t stream = 0;
  double *b2[3] = {0, 0, 0};
  double *part = 0;
  double *scalars = 0;
  double *tab = 0;           /* theta per pencil point (k_solve_zpencil) */
  float *tabf = 0;           /* the same in FP32, for the FP32 mesh      */
  long tab_n = 0;
  bool tab_ok = false;       /* tab holds tab_prm's and tab_zp's theta   */
  RecipParams tab_prm;
  ZPencilBox tab_zp;
  bool single = false;       /* FP32 mesh (nonbond_precision = MIXED) */
  bool built = false;
  std::string err;

  bool fail(const std::string &what, cudaError_t e)
  {
    err = what + ": " + cudaGetErrorString(e);
    return false;
  }
  bool launched(const char *what)
  {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fail(what, e);
    return true;
  }
  void release()
  {
    for (int k = 0; k < 3; ++k) { if (b2[k]) cudaFree(b2[k]); b2[k] = 0; }
    if (part) cudaFree(part);
    if (scalars) cudaFree(scalars);
    if (tab) cudaFree(tab);
    if (tabf) cudaFree(tabf);
    part = scalars = 0;
    tab = 0;
    tabf = 0;
    tab_n = 0;
    tab_ok = false;
    built = false;
  }
  bool upload_b2()
  {
    const double bs_fact = bs_fact_of(prm.nbs);
    for (int k = 0; k < 3; ++k) {
      std::vector<double> h;
      build_b2(prm.nbs, prm.N[k], bs_fact, h);
      /* Synchronous and on the legacy stream, but only ever from build,
       * before any work of this component is queued on `stream`. */
      const cudaError_t e = cudaMemcpy(b2[k], h.data(), h.size() * sizeof(double),
                                       cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return fail("b2 upload", e);
    }
    return true;
  }
};

RecipDevice::RecipDevice() : impl_(new Impl) {}
RecipDevice::~RecipDevice() { destroy(); delete impl_; }

void RecipDevice::destroy() { impl_->release(); }

bool RecipDevice::build(const RecipParams &p, void *stream, bool single)
{
  impl_->release();
  impl_->single = single;
  impl_->err.clear();
  if (p.nbs < 2 || p.nbs > kMaxOrder) {
    impl_->err = "the B-spline order is outside [2,8]";
    return false;
  }
  for (int k = 0; k < 3; ++k)
    if (p.N[k] <= 0) {
      impl_->err = "a mesh axis is not positive";
      return false;
    }
  impl_->prm = p;
  impl_->stream = (cudaStream_t)stream;
  cudaError_t e;
  for (int k = 0; k < 3; ++k) {
    e = cudaMalloc((void **)&impl_->b2[k], (std::size_t)p.N[k] * sizeof(double));
    if (e != cudaSuccess) { impl_->release(); return impl_->fail("b2 alloc", e); }
  }
  e = cudaMalloc((void **)&impl_->part,
                 (std::size_t)kMaxSolveBlocks * kScalars * sizeof(double));
  if (e != cudaSuccess) { impl_->release(); return impl_->fail("partials alloc", e); }
  e = cudaMalloc((void **)&impl_->scalars, kScalars * sizeof(double));
  if (e != cudaSuccess) { impl_->release(); return impl_->fail("scalars alloc", e); }
  /* Defined before any solve, so a rank with an empty pencil reads zeros. */
  e = cudaMemset(impl_->scalars, 0, kScalars * sizeof(double));
  if (e != cudaSuccess) { impl_->release(); return impl_->fail("scalars init", e); }
  if (!impl_->upload_b2()) { impl_->release(); return false; }
  impl_->built = true;
  return true;
}

bool RecipDevice::rebox(const RecipParams &p)
{
  if (!impl_->built) { impl_->err = "rebox before build"; return false; }
  for (int k = 0; k < 3; ++k)
    if (p.N[k] != impl_->prm.N[k]) {
      impl_->err = "rebox changes the mesh";
      return false;
    }
  if (p.nbs != impl_->prm.nbs) {
    impl_->err = "rebox changes the B-spline order";
    return false;
  }
  impl_->prm = p;
  return true;
}

bool RecipDevice::spread(const double *coord, const double *charge,
                         long natoms, long pitch, const BrickBox &brick,
                         void *words)
{
  if (!impl_->built) { impl_->err = "spread before build"; return false; }
  if (const char *why = spread_gather_admissible(impl_->prm)) {
    impl_->err = std::string("spread refused: ") + why;
    return false;
  }
  if (natoms < 0 || (pitch < 0 ? -pitch : pitch) < natoms) { impl_->err = "spread: bad atom count or pitch"; return false; }
  if (natoms == 0 || brick.cells() <= 0) return true;
  const int nbs = impl_->prm.nbs, per = lane_atoms(nbs);
  if (impl_->single) {
    /* orders 4 and 6 at compile time; fixed-point sums give the same bits either way */
    auto *const k = nbs == 4 ? k_spread_brick_tile<4>
                  : nbs == 6 ? k_spread_brick_tile<6> : k_spread_brick_tile<0>;
    S4PME_LAUNCH(k, blocks_for(natoms, kTileAtoms, 1 << 30),
                 per * nbs, impl_->stream)(coord, charge, natoms, pitch,
                                           impl_->prm, brick,
                                           (BrickSum<float>::W *)words);
  } else
    S4PME_LAUNCH(k_spread_brick_lanes<double>, blocks_for(natoms, per, 1 << 30),
                 per * nbs, impl_->stream)(coord, charge, natoms, pitch,
                                           impl_->prm, brick,
                                           (BrickSum<double>::W *)words);
  return impl_->launched("spread");
}

bool RecipDevice::fold(void *words, long cells, void *mesh,
                       const FoldSelf *self)
{
  if (!impl_->built) { impl_->err = "fold before build"; return false; }
  if (cells <= 0) return true;
  FoldSelf none;
  std::memset(&none, 0, sizeof none);
  const FoldSelf &fs = self ? *self : none;
  if (fs.n < 0 || fs.n > kMaxFoldSelf ||
      (fs.n > 0 && (fs.src == 0 || fs.d0 <= 0 || fs.d1 <= 0 || cells > 0x7fffffffL))) {
    impl_->err = "fold: bad self boxes";
    return false;
  }
  const int fb = blocks_for(cells, kBlock, 65535);
  if (impl_->single)
    k_fold_brick<float, BrickSum<float>::W, BrickSum<float>::R>
        <<<fb, kBlock, 0, impl_->stream>>>((BrickSum<float>::W *)words, cells,
                                           (float *)mesh, fs);
  else
    k_fold_brick<double, BrickSum<double>::W, BrickSum<double>::R>
        <<<fb, kBlock, 0, impl_->stream>>>((BrickSum<double>::W *)words, cells,
                                           (double *)mesh, fs);
  return impl_->launched("fold");
}

bool RecipDevice::solve(void *zpencil, const ZPencilBox &zp, bool scalars)
{
  if (!impl_->built) { impl_->err = "solve before build"; return false; }
  const long n = zp.points();
  if (n <= 0) {
    const cudaError_t e = cudaMemsetAsync(impl_->scalars, 0,
                                          kScalars * sizeof(double),
                                          impl_->stream);
    if (e != cudaSuccess) return impl_->fail("solve: empty pencil", e);
    return true;
  }
  const int nblk = blocks_for(n, kBlock, kMaxSolveBlocks);
  Impl &m = *impl_;
  if (n > m.tab_n) {
    if (m.tab) cudaFree(m.tab);
    if (m.tabf) cudaFree(m.tabf);
    m.tab = 0;
    m.tabf = 0;
    m.tab_n = 0;
    m.tab_ok = false;
    cudaError_t e = cudaMalloc((void **)&m.tab, (std::size_t)n * sizeof(double));
    if (e != cudaSuccess) return m.fail("solve: theta table alloc", e);
    if (m.single) {
      e = cudaMalloc((void **)&m.tabf, (std::size_t)n * sizeof(float));
      if (e != cudaSuccess) return m.fail("solve: FP32 theta table alloc", e);
    }
    m.tab_n = n;
  }
  /* theta depends on the box (gfact), alpha (theta_fact) and the pencil;
   * the mesh and order are fixed from build (rebox refuses to change them). */
  if (m.tab_ok)
    for (int k = 0; k < 3; ++k)
      if (m.tab_prm.gfact[k] != m.prm.gfact[k]) m.tab_ok = false;
  if (m.tab_ok &&
      (m.tab_prm.theta_fact != m.prm.theta_fact || m.tab_zp.kx0 != zp.kx0 ||
       m.tab_zp.nkx != zp.nkx || m.tab_zp.ky0 != zp.ky0 ||
       m.tab_zp.nky != zp.nky || m.tab_zp.nz != zp.nz ||
       m.tab_zp.kx_fastest != zp.kx_fastest))
    m.tab_ok = false;
  const bool load = m.tab_ok;
#define S4PME_SOLVE(C, S, Th)                                                 \
  k_solve_zpencil<C, S, Th><<<nblk, kBlock, 0, m.stream>>>(                   \
      (C *)zpencil, zp, m.prm, m.b2[0], m.b2[1], m.b2[2], m.tab, m.tabf, m.part)
  if (m.single) {
    if (load && scalars)       S4PME_SOLVE(cufftComplex, true, kThetaLoad);
    else if (load)             S4PME_SOLVE(cufftComplex, false, kThetaLoad);
    else if (scalars)          S4PME_SOLVE(cufftComplex, true, kThetaStore);
    else                       S4PME_SOLVE(cufftComplex, false, kThetaStore);
  } else {
    if (load && scalars)       S4PME_SOLVE(cufftDoubleComplex, true, kThetaLoad);
    else if (load)             S4PME_SOLVE(cufftDoubleComplex, false, kThetaLoad);
    else if (scalars)          S4PME_SOLVE(cufftDoubleComplex, true, kThetaStore);
    else                       S4PME_SOLVE(cufftDoubleComplex, false, kThetaStore);
  }
#undef S4PME_SOLVE
  if (!m.launched("solve")) return false;
  if (!load) {
    m.tab_ok = true;
    m.tab_prm = m.prm;
    m.tab_zp = zp;
  }
  if (!scalars) return true;
  S4PME_LAUNCH(k_sum_scalars, 1, 32 * kScalars, impl_->stream)(impl_->part, nblk,
                                                     impl_->scalars);
  return impl_->launched("solve finish");
}

bool RecipDevice::gather(const double *coord, const double *charge,
                         long natoms, long pitch, const BrickBox &brick,
                         const void *brick_buf, double *force)
{
  if (!impl_->built) { impl_->err = "gather before build"; return false; }
  if (const char *why = spread_gather_admissible(impl_->prm)) {
    impl_->err = std::string("gather refused: ") + why;
    return false;
  }
  if (natoms < 0 || (pitch < 0 ? -pitch : pitch) < natoms) { impl_->err = "gather: bad atom count or pitch"; return false; }
  if (natoms == 0) return true;
  const int nbs = impl_->prm.nbs;
  if (impl_->single) {
    auto *const k = nbs == 4 ? k_gather_brick<float, 4>
                  : nbs == 6 ? k_gather_brick<float, 6> : k_gather_brick<float, 0>;
    S4PME_LAUNCH(k, blocks_for(natoms, kBlock, 1 << 30),
                 kBlock, impl_->stream)(coord, charge, natoms, pitch, impl_->prm,
                                        brick, (const float *)brick_buf, force);
  } else {
    auto *const k = nbs == 4 ? k_gather_brick<double, 4>
                  : nbs == 6 ? k_gather_brick<double, 6> : k_gather_brick<double, 0>;
    S4PME_LAUNCH(k, blocks_for(natoms, kBlock, 1 << 30),
                 kBlock, impl_->stream)(coord, charge, natoms, pitch, impl_->prm,
                                        brick, (const double *)brick_buf, force);
  }
  return impl_->launched("gather");
}

const double *RecipDevice::scalars_device() const { return impl_->scalars; }

const RecipParams &RecipDevice::params() const { return impl_->prm; }
const char *RecipDevice::last_error() const { return impl_->err.c_str(); }

} /* namespace genesis_native_s4pme */
