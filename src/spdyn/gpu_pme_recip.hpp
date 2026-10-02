/* Distributed PME reciprocal component, host-visible half.
 *
 * Around the pencil FFT (gpu_pencil_fft.hpp):
 *   spread   atoms -> fixed-point sums over this rank's brick
 *   fold     sums -> mesh, before the forward FFT
 *   solve    the Z pencil in place, plus energy and virial partials
 *   gather   this rank's convolved brick -> per-atom forces
 *
 * The brick sums are integers, so bricks of neighbouring ranks may be added in
 * any order and give the whole-mesh spread bit for bit.  A stencil cell outside
 * the brick poisons the sums (NaN on every fold).  The convolved mesh is
 * UNNORMALISED (no 1/(Nx Ny Nz)); vol_fact4 in the gather makes the force the
 * gradient of the solve's energy.
 *
 * Every call is asynchronous on the stream given to build(); nothing here
 * synchronises.
 */
#ifndef GENESIS_NATIVE_S4PME_RECIP_HPP
#define GENESIS_NATIVE_S4PME_RECIP_HPP

#include "gpu_pme_recip_numerics.hpp"

#include <cstddef>

namespace genesis_native_s4pme {

/* This rank's brick: global cells (x0 + i) mod Nx for i in [0,nx), alike in
 * y and z, stored [z][y][x] with x fastest.  0 <= x0 < Nx and nx <= Nx; a
 * brick may wrap past the mesh edge. */
struct BrickBox {
  long x0, y0, z0;
  long nx, ny, nz;
  long cells() const { return nx * ny * nz; }
};

/* Part of a forward brick move that stays on the rank: up to kMaxFoldSelf
 * boxes of brick sums that land on this rank's pencil, added by fold as it
 * reads the pencil's sums.  Pencil cell s = z*d0 + y*d1 + x; a box covers
 * [z0,z0+nz) x [y0,y0+ny) x [x0,x0+nx) and reads brick sums at
 * off + (z-z0)*s0 + (y-y0)*s1 + (x-x0)*s2. */
const int kMaxFoldSelf = 8;
struct FoldSelf {
  int n;                       /* boxes; 0 = none */
  const void *src;             /* the brick's sums */
  long d0, d1;
  struct Box {
    long off, s0, s1, s2;
    int z0, y0, x0, nz, ny, nx;
  } box[kMaxFoldSelf];
};

/* This rank's Z pencil: kx in [kx0,kx0+nkx) of the half spectrum
 * [0, Nx/2+1), ky in [ky0,ky0+nky), every kz in [0,Nz); stored [ky][kx][kz]
 * with kz fastest.  Empty is legal.  kx_fastest: stored [kz][ky][kx] instead
 * (the whole spectrum of a single-rank 3-D transform). */
struct ZPencilBox {
  long kx0, nkx;
  long ky0, nky;
  long nz;
  bool kx_fastest;
  long points() const { return nkx * nky * nz; }
};

/* The seven scalars, in this order: energy, then the symmetric virial xx, xy,
 * xz, yy, yz, zz.  They carry vol_fact2 (including the factor 1/2), so they
 * are in the caller's energy unit. */
enum { kScalars = 7 };

class RecipDevice {
public:
  RecipDevice();
  ~RecipDevice();
  RecipDevice(const RecipDevice &) = delete;
  RecipDevice &operator=(const RecipDevice &) = delete;

  /* Upload b2 for the three axes, allocate the solve's partial-sum workspace
   * and bind to `stream`.  Returns false and leaves nothing allocated on
   * failure.  single: float mesh and spectrum, FP32 B-spline weights and
   * gather sums (nonbond_precision = MIXED); the spread still accumulates in
   * 32-bit fixed point (2^-24) and the solve's energy and virial in FP64. */
  bool build(const RecipParams &p, void *stream, bool single = false);
  /* New box on the same mesh and order: b2 is unchanged, nothing is
   * re-uploaded. */
  bool rebox(const RecipParams &p);
  void destroy();

  /* Add every atom's stencil cells to `words`, the brick's fixed-point sums
   * (one 64-bit word per cell, 32-bit for single), which must hold zero or
   * earlier sums.  coord is SoA: x at [0,pitch), y at [pitch,2 pitch), z at
   * [2 pitch,..).  natoms == 0 is legal. */
  bool spread(const double *coord, const double *charge, long natoms,
              long pitch, const BrickBox &brick, void *words);

  /* `cells` sums -> `mesh` (double, float for single); the sums are zeroed
   * in the same pass.  `self`, when given, adds its boxes' sums first. */
  bool fold(void *words, long cells, void *mesh, const FoldSelf *self = 0);

  /* Spectral pass on this rank's Z pencil, in place; the seven partial
   * scalars stay on the device (scalars_device()).  scalars == false skips
   * them and keeps the last values. */
  bool solve(void *zpencil, const ZPencilBox &zp, bool scalars = true);

  /* Per-atom force from this rank's brick of the convolved mesh.  force is
   * SoA with the same pitch; entries [0,natoms) are OVERWRITTEN, padding is
   * not touched. */
  bool gather(const double *coord, const double *charge, long natoms,
              long pitch, const BrickBox &brick, const void *brick_buf,
              double *force);

  /* The seven partials of the last solve, on the device. */
  const double *scalars_device() const;

  const RecipParams &params() const;
  const char *last_error() const;

private:
  struct Impl;
  Impl *impl_;
};

} /* namespace genesis_native_s4pme */

#endif
