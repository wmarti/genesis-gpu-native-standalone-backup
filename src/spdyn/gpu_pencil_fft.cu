/*
 * gpu_pencil_fft.cu : pencil FFT pipeline, device half.
 *
 * Three batched one-dimensional transforms, each along the axis that is
 * stride 1 in its own pencil, with idist == odist == its transform length:
 *
 *   x  1-D CUFFT_D2Z  batch nyA*nzB, idist Nx, odist Nxh   real -> complex
 *   y  1-D CUFFT_Z2Z  batch nxhA*nzB, idist Ny, odist Ny   complex, in place
 *   z  1-D CUFFT_Z2Z  batch nxhA*nyB, idist Nz, odist Nz   complex, in place
 *
 * with the x plans mirrored for the inverse (CUFFT_Z2D, idist Nxh, odist Nx).
 * cuFFT is unnormalised: a forward followed by an inverse scales the field by
 * Nx*Ny*Nz, and the caller applies fft_inverse_scale().
 *
 * Every plan is created with cufftSetAutoAllocation(h, 0), reports its own
 * work size, and one allocation of the MAXIMUM is shared by all of them.  That
 * is sound only because the plans run one after another on one stream.
 *
 * An axis with no lines is a no-op: a rank owning no cells there runs the
 * same code path as every other rank.
 */
#include "gpu_pencil_fft.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>
#include <string>

namespace genesis_native_s4fft {

namespace {

std::string cufft_message(const char *what, cufftResult r)
{
  char buf[128];
  std::snprintf(buf, sizeof buf, "%d", (int)r);
  return std::string("s4fft ") + what + ": cufft error " + buf;
}

} /* namespace */

struct PencilFftPlans::Impl {
  FftAxes axes;
  cufftHandle px_f = 0, px_i = 0, py = 0, pz = 0;
  bool have_px = false, have_py = false, have_pz = false;
  void *d_work = 0;
  std::size_t requested = 0;
  std::size_t allocated = 0;
  int nplans = 0;
  cudaStream_t stream = 0;
  bool single = false;
  std::string err;

  void release()
  {
    if (have_px) { cufftDestroy(px_f); cufftDestroy(px_i); }
    if (have_py) cufftDestroy(py);
    if (have_pz) cufftDestroy(pz);
    px_f = px_i = py = pz = 0;
    have_px = have_py = have_pz = false;
    if (d_work) cudaFree(d_work);
    d_work = 0;
  }
};

PencilFftPlans::PencilFftPlans() : impl_(new Impl()) {}

PencilFftPlans::~PencilFftPlans()
{
  destroy();
  delete impl_;
  impl_ = 0;
}

void PencilFftPlans::destroy()
{
  if (impl_ == 0) return;
  impl_->release();
  impl_->requested = 0;
  impl_->allocated = 0;
  impl_->nplans = 0;
}

bool PencilFftPlans::build(const FftAxes &axes, void *stream, bool single)
{
  if (impl_ == 0) return false;
  impl_->release();
  impl_->axes = axes;
  impl_->stream = (cudaStream_t)stream;
  impl_->single = single;
  impl_->err.clear();
  const cufftType t_r2c = single ? CUFFT_R2C : CUFFT_D2Z;
  const cufftType t_c2r = single ? CUFFT_C2R : CUFFT_Z2D;
  const cufftType t_c2c = single ? CUFFT_C2C : CUFFT_Z2Z;

  std::size_t wmax = 0;
  /* cufftMakePlanMany takes a non-const `int *n`, so the transform lengths
   * cannot be const. */
  int nX = (int)axes.nx, nY = (int)axes.ny, nZ = (int)axes.nz;
  const int nXh = (int)axes.nxh;
  std::size_t w = 0;
  cufftResult r = CUFFT_SUCCESS;

  if (axes.have_x()) {
    r = cufftCreate(&impl_->px_f);
    if (r == CUFFT_SUCCESS) r = cufftSetAutoAllocation(impl_->px_f, 0);
    if (r == CUFFT_SUCCESS)
      r = cufftMakePlanMany(impl_->px_f, 1, &nX, NULL, 1, nX, NULL, 1, nXh,
                            t_r2c, (int)axes.batch_x, &w);
    if (r != CUFFT_SUCCESS) { impl_->err = cufft_message("plan x forward", r); impl_->release(); return false; }
    ++impl_->nplans; wmax = w > wmax ? w : wmax;

    r = cufftCreate(&impl_->px_i);
    if (r == CUFFT_SUCCESS) r = cufftSetAutoAllocation(impl_->px_i, 0);
    if (r == CUFFT_SUCCESS)
      r = cufftMakePlanMany(impl_->px_i, 1, &nX, NULL, 1, nXh, NULL, 1, nX,
                            t_c2r, (int)axes.batch_x, &w);
    if (r != CUFFT_SUCCESS) { impl_->err = cufft_message("plan x inverse", r); impl_->release(); return false; }
    ++impl_->nplans; wmax = w > wmax ? w : wmax;
    impl_->have_px = true;
  }

  if (axes.have_y()) {
    r = cufftCreate(&impl_->py);
    if (r == CUFFT_SUCCESS) r = cufftSetAutoAllocation(impl_->py, 0);
    if (r == CUFFT_SUCCESS)
      r = cufftMakePlanMany(impl_->py, 1, &nY, NULL, 1, nY, NULL, 1, nY,
                            t_c2c, (int)axes.batch_y, &w);
    if (r != CUFFT_SUCCESS) { impl_->err = cufft_message("plan y", r); impl_->release(); return false; }
    ++impl_->nplans; wmax = w > wmax ? w : wmax;
    impl_->have_py = true;
  }

  if (axes.have_z()) {
    r = cufftCreate(&impl_->pz);
    if (r == CUFFT_SUCCESS) r = cufftSetAutoAllocation(impl_->pz, 0);
    if (r == CUFFT_SUCCESS)
      r = cufftMakePlanMany(impl_->pz, 1, &nZ, NULL, 1, nZ, NULL, 1, nZ,
                            t_c2c, (int)axes.batch_z, &w);
    if (r != CUFFT_SUCCESS) { impl_->err = cufft_message("plan z", r); impl_->release(); return false; }
    ++impl_->nplans; wmax = w > wmax ? w : wmax;
    impl_->have_pz = true;
  }

  impl_->requested = wmax;
  if (impl_->nplans == 0) {
    /* Every axis is empty: a rank that owns nothing; nothing to allocate or run. */
    impl_->allocated = 0;
    return true;
  }

  if (cudaMalloc(&impl_->d_work, wmax) != cudaSuccess) {
    impl_->err = "s4fft work area: cudaMalloc failed";
    impl_->release();
    return false;
  }
  impl_->allocated = wmax;

  /* One block, handed to all of them.  Valid because they never overlap. */
  if (impl_->have_px) {
    if (cufftSetWorkArea(impl_->px_f, impl_->d_work) != CUFFT_SUCCESS ||
        cufftSetWorkArea(impl_->px_i, impl_->d_work) != CUFFT_SUCCESS ||
        cufftSetStream(impl_->px_f, impl_->stream) != CUFFT_SUCCESS ||
        cufftSetStream(impl_->px_i, impl_->stream) != CUFFT_SUCCESS) {
      impl_->err = "s4fft work area: the x plans refused it";
      impl_->release();
      return false;
    }
  }
  if (impl_->have_py) {
    if (cufftSetWorkArea(impl_->py, impl_->d_work) != CUFFT_SUCCESS ||
        cufftSetStream(impl_->py, impl_->stream) != CUFFT_SUCCESS) {
      impl_->err = "s4fft work area: the y plan refused it";
      impl_->release();
      return false;
    }
  }
  if (impl_->have_pz) {
    if (cufftSetWorkArea(impl_->pz, impl_->d_work) != CUFFT_SUCCESS ||
        cufftSetStream(impl_->pz, impl_->stream) != CUFFT_SUCCESS) {
      impl_->err = "s4fft work area: the z plan refused it";
      impl_->release();
      return false;
    }
  }
  return true;
}

bool PencilFftPlans::build_3d(long nx, long ny, long nz, void *stream,
                              bool single)
{
  if (impl_ == 0) return false;
  impl_->release();
  impl_->requested = impl_->allocated = 0;
  impl_->nplans = 0;
  impl_->stream = (cudaStream_t)stream;
  impl_->single = single;
  impl_->err.clear();
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    impl_->err = "s4fft build_3d: a mesh axis is not positive";
    return false;
  }
  /* Row-major lengths, slowest first: the brick's [z][y][x]; the R2C halves x. */
  int n[3] = {(int)nz, (int)ny, (int)nx};
  std::size_t wf = 0, wi = 0;
  cufftResult r = cufftCreate(&impl_->px_f);
  if (r == CUFFT_SUCCESS) {
    r = cufftCreate(&impl_->px_i);
    if (r != CUFFT_SUCCESS) cufftDestroy(impl_->px_f);
  }
  if (r != CUFFT_SUCCESS) {
    impl_->err = cufft_message("plan 3d create", r);
    return false;
  }
  impl_->have_px = true;
  r = cufftSetAutoAllocation(impl_->px_f, 0);
  if (r == CUFFT_SUCCESS) r = cufftSetAutoAllocation(impl_->px_i, 0);
  if (r == CUFFT_SUCCESS)
    r = cufftMakePlanMany(impl_->px_f, 3, n, NULL, 1, 0, NULL, 1, 0,
                          single ? CUFFT_R2C : CUFFT_D2Z, 1, &wf);
  if (r == CUFFT_SUCCESS)
    r = cufftMakePlanMany(impl_->px_i, 3, n, NULL, 1, 0, NULL, 1, 0,
                          single ? CUFFT_C2R : CUFFT_Z2D, 1, &wi);
  if (r != CUFFT_SUCCESS) {
    impl_->err = cufft_message("plan 3d", r);
    impl_->release();
    return false;
  }
  impl_->nplans = 2;
  impl_->requested = wf > wi ? wf : wi;
  if (impl_->requested > 0 &&
      cudaMalloc(&impl_->d_work, impl_->requested) != cudaSuccess) {
    impl_->err = "s4fft work area: cudaMalloc failed";
    impl_->release();
    return false;
  }
  impl_->allocated = impl_->requested;
  if (cufftSetWorkArea(impl_->px_f, impl_->d_work) != CUFFT_SUCCESS ||
      cufftSetWorkArea(impl_->px_i, impl_->d_work) != CUFFT_SUCCESS ||
      cufftSetStream(impl_->px_f, impl_->stream) != CUFFT_SUCCESS ||
      cufftSetStream(impl_->px_i, impl_->stream) != CUFFT_SUCCESS) {
    impl_->err = "s4fft work area: the 3d plans refused it";
    impl_->release();
    return false;
  }
  return true;
}

#define S4FFT_EXEC(what, call)                                            \
  do {                                                                      \
    if (impl_ == 0) return false;                                           \
    const cufftResult r_ = (call);                                          \
    if (r_ != CUFFT_SUCCESS) { impl_->err = cufft_message(what, r_); return false; } \
    return true;                                                            \
  } while (0)

bool PencilFftPlans::exec_forward_x(const void *real, void *complex_out)
{
  if (impl_ == 0) return false;
  if (!impl_->have_px) return true;   /* an empty partition has nothing to do */
  S4FFT_EXEC("fft x forward",
             impl_->single
                 ? cufftExecR2C(impl_->px_f, (cufftReal *)real,
                                (cufftComplex *)complex_out)
                 : cufftExecD2Z(impl_->px_f, (cufftDoubleReal *)real,
                                (cufftDoubleComplex *)complex_out));
}

bool PencilFftPlans::exec_forward_y(void *complex_inout)
{
  if (impl_ == 0) return false;
  if (!impl_->have_py) return true;
  S4FFT_EXEC("fft y forward",
             impl_->single
                 ? cufftExecC2C(impl_->py, (cufftComplex *)complex_inout,
                                (cufftComplex *)complex_inout, CUFFT_FORWARD)
                 : cufftExecZ2Z(impl_->py, (cufftDoubleComplex *)complex_inout,
                                (cufftDoubleComplex *)complex_inout, CUFFT_FORWARD));
}

bool PencilFftPlans::exec_forward_z(void *complex_inout)
{
  if (impl_ == 0) return false;
  if (!impl_->have_pz) return true;
  S4FFT_EXEC("fft z forward",
             impl_->single
                 ? cufftExecC2C(impl_->pz, (cufftComplex *)complex_inout,
                                (cufftComplex *)complex_inout, CUFFT_FORWARD)
                 : cufftExecZ2Z(impl_->pz, (cufftDoubleComplex *)complex_inout,
                                (cufftDoubleComplex *)complex_inout, CUFFT_FORWARD));
}

bool PencilFftPlans::exec_inverse_z(void *complex_inout)
{
  if (impl_ == 0) return false;
  if (!impl_->have_pz) return true;
  S4FFT_EXEC("fft z inverse",
             impl_->single
                 ? cufftExecC2C(impl_->pz, (cufftComplex *)complex_inout,
                                (cufftComplex *)complex_inout, CUFFT_INVERSE)
                 : cufftExecZ2Z(impl_->pz, (cufftDoubleComplex *)complex_inout,
                                (cufftDoubleComplex *)complex_inout, CUFFT_INVERSE));
}

bool PencilFftPlans::exec_inverse_y(void *complex_inout)
{
  if (impl_ == 0) return false;
  if (!impl_->have_py) return true;
  S4FFT_EXEC("fft y inverse",
             impl_->single
                 ? cufftExecC2C(impl_->py, (cufftComplex *)complex_inout,
                                (cufftComplex *)complex_inout, CUFFT_INVERSE)
                 : cufftExecZ2Z(impl_->py, (cufftDoubleComplex *)complex_inout,
                                (cufftDoubleComplex *)complex_inout, CUFFT_INVERSE));
}

bool PencilFftPlans::exec_inverse_x(const void *complex_in, void *real_out)
{
  if (impl_ == 0) return false;
  if (!impl_->have_px) return true;
  S4FFT_EXEC("fft x inverse",
             impl_->single
                 ? cufftExecC2R(impl_->px_i, (cufftComplex *)complex_in,
                                (cufftReal *)real_out)
                 : cufftExecZ2D(impl_->px_i, (cufftDoubleComplex *)complex_in,
                                (cufftDoubleReal *)real_out));
}

#undef S4FFT_EXEC

int PencilFftPlans::plans() const { return impl_ == 0 ? 0 : impl_->nplans; }

std::size_t PencilFftPlans::work_bytes_requested() const
{
  return impl_ == 0 ? 0 : impl_->requested;
}

std::size_t PencilFftPlans::work_bytes_allocated() const
{
  return impl_ == 0 ? 0 : impl_->allocated;
}

const char *PencilFftPlans::last_error() const
{
  return impl_ == 0 ? "" : impl_->err.c_str();
}

} /* namespace genesis_native_s4fft */
