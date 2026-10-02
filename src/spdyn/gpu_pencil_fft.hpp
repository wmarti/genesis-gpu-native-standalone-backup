/*
 * gpu_pencil_fft.hpp : pencil FFT pipeline, host-visible half.
 *
 * Three one-dimensional batched cuFFT transforms, each along the axis that is
 * stride 1 in its pencil, with idist == odist == the transform length:
 *
 *   X  D2Z  real [z][y][x]      -> complex [z][y][xh]   batch nyA*nzB
 *   Y  Z2Z  complex [z][xh][y]  -> in place              batch nxhA*nzB
 *   Z  Z2Z  complex [xh][y][z]  -> in place              batch nxhA*nyB
 *
 * and back, the last being Z2D followed by the 1/(Nx*Ny*Nz) rescale that
 * cuFFT's unnormalised convention needs.  The pencil-to-pencil moves are the
 * exchange plans of gpu_mesh_exchange_session.hpp.
 *
 * Work areas are caller-owned (cufftSetAutoAllocation(h, 0)); ONE allocation
 * of the maximum size serves all plans.  This is sound only because the plans
 * run strictly one at a time on one stream.
 */
#ifndef GENESIS_NATIVE_S4FFT_PENCIL_FFT_HPP
#define GENESIS_NATIVE_S4FFT_PENCIL_FFT_HPP

#include "gpu_mesh_exchange_plan.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace genesis_native_s4fft {

using count_t = genesis_native_pencil::count_t;

/* Which axis each pencil transforms and how many lines it has.  Every field
 * comes from an ExchangePlan, so the FFT layer never re-derives a pencil size.
 * x_elements is the x->y plan's source (the X pencil), y_elements the y->z
 * plan's source (the Y pencil), z_elements the y->z plan's target (the Z
 * pencil). */
struct FftAxes {
  long nx, nxh, batch_x;   /* R2C along x: real [z][y][x] -> complex [z][y][xh] */
  long ny, batch_y;        /* Z2Z along y, in place, in the Y pencil           */
  long nz, batch_z;        /* Z2Z along z, in place, in the Z pencil           */

  /* The element counts the batches were derived from.  The Y and Z pencils
   * use the rank-local xh extent, not the global nxh, so the x batch is
   * x_elements / nxh but the y batch is y_elements / ny. */
  count_t x_elements, y_elements, z_elements;

  bool have_x() const { return batch_x > 0; }
  bool have_y() const { return batch_y > 0; }
  bool have_z() const { return batch_z > 0; }

  /* Elements each pencil holds (the buffer size). */
  count_t x_complex_elements() const;
  count_t y_complex_elements() const;
  count_t z_complex_elements() const;

  /* The real buffer the R2C input lives in: 2*nxh doubles per line hold
   * either the Nx reals in or the Nxh complex out. */
  count_t real_elements() const;
};

/* Derive the axes.  Throws on a combination the plans cannot have
 * produced. */
FftAxes fft_axes(long nx, long ny, long nz, count_t x_elements,
                 count_t y_elements, count_t z_elements);

/* The cuFFT half.  Defined in gpu_pencil_fft.cu (needs CUDA); the declaration
 * is here so host code can include this header without a device. */
class PencilFftPlans {
public:
  PencilFftPlans();
  ~PencilFftPlans();
  PencilFftPlans(const PencilFftPlans &) = delete;
  PencilFftPlans &operator=(const PencilFftPlans &) = delete;

  /* Plan every transform the axes need, allocate ONE work area of the
   * maximum size, hand it to all plans and bind them to `stream`.  Returns
   * false and leaves nothing allocated on failure.  single: FP32 transforms
   * on float / float2 buffers (nonbond_precision = MIXED). */
  bool build(const FftAxes &axes, void *stream, bool single = false);

  /* One rank holds the whole mesh: one 3-D real-to-complex plan and its
   * inverse on the brick itself, so no pencil move runs.  real [z][y][x] <->
   * complex [z][y][xh] (xh = nx/2+1), out of place.  exec_forward_x and
   * exec_inverse_x run the whole transform (the inverse may overwrite the
   * spectrum); the y and z calls are no-ops. */
  bool build_3d(long nx, long ny, long nz, void *stream, bool single = false);

  void destroy();

  /* Plans made, the largest work size any asked for, and the bytes actually
   * allocated (one allocation, so the maximum, not the sum). */
  int plans() const;
  std::size_t work_bytes_requested() const;
  std::size_t work_bytes_allocated() const;

  /* Forward: D2Z over x, Z2Z over y, Z2Z over z.  Reverse: the same
   * backwards, Z2D last.  A zero-length axis is a no-op. */
  bool exec_forward_x(const void *real, void *complex_out);
  bool exec_forward_y(void *complex_inout);
  bool exec_forward_z(void *complex_inout);
  bool exec_inverse_z(void *complex_inout);
  bool exec_inverse_y(void *complex_inout);
  bool exec_inverse_x(const void *complex_in, void *real_out);

  const char *last_error() const;

private:
  struct Impl;
  Impl *impl_;
};

inline count_t FftAxes::x_complex_elements() const
{
  /* The X pencil is [z][y][xh]; every line is nxh long. */
  return (count_t)nxh * (count_t)batch_x;
}

inline count_t FftAxes::y_complex_elements() const { return y_elements; }
inline count_t FftAxes::z_complex_elements() const { return z_elements; }

inline count_t FftAxes::real_elements() const
{
  /* Nx reals in, Nxh complex out: twice Nxh doubles per line holds either. */
  return (count_t)(2 * nxh) * (count_t)batch_x;
}

inline FftAxes fft_axes(long nx, long ny, long nz, count_t x_elements,
                        count_t y_elements, count_t z_elements)
{
  if (nx <= 0 || ny <= 0 || nz <= 0)
    throw std::invalid_argument("s4fft fft_axes: a mesh axis is not positive");

  /* nx need not be even: the r2c output length is nx/2 + 1 by integer
   * division, so an odd nx has no Nyquist slot. */
  FftAxes a;
  a.nx = nx;
  a.nxh = nx / 2 + 1;
  a.ny = ny;
  a.nz = nz;

  a.x_elements = x_elements;
  a.y_elements = y_elements;
  a.z_elements = z_elements;

  /* Batch counts come from the plan layer's element counts: each transform
   * runs along the axis that is stride 1 in its pencil, so the batch is the
   * element count divided by that axis length (x by the global nxh, y and z
   * by their own axis).  A count that does not divide is refused. */
  if (x_elements % (count_t)a.nxh != 0)
    throw std::invalid_argument("s4fft fft_axes: the X pencil is not a whole "
                                "number of r2c lines");
  a.batch_x = (long)(x_elements / (count_t)a.nxh);

  if (y_elements % (count_t)ny != 0)
    throw std::invalid_argument("s4fft fft_axes: the Y pencil is not a whole "
                                "number of y lines");
  a.batch_y = (long)(y_elements / (count_t)ny);

  if (z_elements % (count_t)nz != 0)
    throw std::invalid_argument("s4fft fft_axes: the Z pencil is not a whole "
                                "number of z lines");
  a.batch_z = (long)(z_elements / (count_t)nz);

  return a;
}

} /* namespace genesis_native_s4fft */

#endif /* GENESIS_NATIVE_S4FFT_PENCIL_FFT_HPP */
