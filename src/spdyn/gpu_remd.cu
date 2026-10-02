/*
 * gpu_remd.cu : device side of a temperature-exchange boundary for a native
 * context that stays resident across the REMD cycle.
 *
 * Both entry points run on the context's stream and touch only owned slots
 * 0 .. num_owned-1 of each component (component c of slot s is at
 * [c*pitch + s]).
 *   gpu_core_remd_rollback  restore velocity and coordinates to the references
 *                           saved by the cycle's extra VV1 (stock coord_vel_ref).
 *   gpu_core_remd_rescale   accepted-exchange velocity rescale: velocity *= factor.
 *                           Values are validated on the device before any write.
 * The factor and the acceptance are the host's; no particle value crosses to
 * the host (validation returns one int).
 */

#include "gpu_core_native.h"
#include "gpu_core_internal.h"

namespace {

template <class S>
__global__ void gcn_kern_remd_rollback(gc_f64 *__restrict__ coord,
                                       const gc_f64 *__restrict__ coord_ref,
                                       S *__restrict__ vel,
                                       const gc_f64 *__restrict__ vel_ref,
                                       gc_f64 *__restrict__ vel_half,
                                       gc_i64 n, gc_i64 pitch)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        for (int c = 0; c < 3; ++c) {
            const gc_i64 s = (gc_i64)c * pitch + i;
            vel_half[s] = vel[s];
            vel[s]      = vel_ref[s];
            coord[s]    = coord_ref[s];
        }
    }
}

/* Worst verdict over the owned velocities: 2 a value is not finite, 1 a
 * scaled value is not finite, 0 none. */
template <class S>
__global__ void gcn_kern_remd_validate(const S *__restrict__ vel,
                                       gc_f64 factor, gc_i64 n, gc_i64 pitch,
                                       int *__restrict__ worst)
{
    int err = 0;
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c) {
            const double v = vel[(gc_i64)c * pitch + i];
            if (!isfinite(v)) err = 2;
            else if (!isfinite((double)(S)(v * factor)) && err == 0) err = 1;
        }
    if (err != 0) atomicMax(worst, err);
}

template <class S>
__global__ void gcn_kern_remd_scale(S *__restrict__ vel, gc_f64 factor,
                                    gc_i64 n, gc_i64 pitch)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x)
        for (int c = 0; c < 3; ++c) vel[(gc_i64)c * pitch + i] *= factor;
}

}  /* anonymous namespace */

extern "C" gc_status gpu_core_remd_rollback(gc_context *ctx)
{
    if (ctx == 0 || ctx->native == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid || !d->step_ready || d->native_steps <= 0)
        return GC_E_STATE;
    if (d->num_owned <= 0) return GC_OK;
    gc_i64 nb = (d->num_owned + 255) / 256;
    if (nb > 1024) nb = 1024;
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_remd_rollback<<<(unsigned)nb, 256, 0, d->stream>>>(
            d->coord, d->coord_ref, d->vel.as<decltype(z)>(), d->vel_ref,
            d->vel_half, d->num_owned, d->pitch);
    });
    return gcn::dev_launched("remd_rollback");
}

/* verdict: 0 applied; 1 a scaled value would overflow; 2 a velocity is not
 * finite.  On a non-zero verdict nothing was written (the adapter's
 * validate-before-mutate rule).  A launch or completion failure after
 * validation returns GC_E_DEVICE: the in-place scale has no rollback, so
 * the caller must stop the run. */
extern "C" gc_status gpu_core_remd_rescale(gc_context *ctx, gc_f64 factor,
                                           gc_i32 *verdict)
{
    if (ctx == 0 || ctx->native == 0 || verdict == 0) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->list_valid || !d->step_ready) return GC_E_STATE;
    *verdict = 0;
    if (d->num_owned <= 0) return GC_OK;

    gc_i64 nb = (d->num_owned + 255) / 256;
    if (nb > 1024) nb = 1024;
    int *worst = 0, h_worst = 0;
    if (cudaMalloc((void **)&worst, sizeof(int)) != cudaSuccess)
        return GC_E_DEVICE;
    cudaError_t e = cudaMemsetAsync(worst, 0, sizeof(int), d->stream);
    if (e == cudaSuccess) {
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_remd_validate<<<(unsigned)nb, 256, 0, d->stream>>>(
                d->vel.as<decltype(z)>(), factor, d->num_owned, d->pitch,
                worst);
        });
        e = cudaMemcpyAsync(&h_worst, worst, sizeof(int),
                            cudaMemcpyDeviceToHost, d->stream);
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(d->stream);
    cudaFree(worst);
    if (e != cudaSuccess) return GC_E_DEVICE;
    if (h_worst != 0) {
        *verdict = (gc_i32)h_worst;
        return GC_OK;
    }
    vf_each(d->vf32, [&](auto z) {
        gcn_kern_remd_scale<<<(unsigned)nb, 256, 0, d->stream>>>(
            d->vel.as<decltype(z)>(), factor, d->num_owned, d->pitch);
    });
    if (cudaStreamSynchronize(d->stream) != cudaSuccess) return GC_E_DEVICE;
    return GC_OK;
}
