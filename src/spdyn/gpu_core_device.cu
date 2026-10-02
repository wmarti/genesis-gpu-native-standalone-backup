/* gpu_core_device.cu : free and total device memory for the allocation plan. */

#include "gpu_core_internal.h"

#include <cuda_runtime.h>

namespace gcn {

gc_status device_meminfo(gc_i64 *bytes_free, gc_i64 *bytes_total,
                         const char **name_out)
{
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        cudaGetLastError();
        if (bytes_free)  *bytes_free  = 0;
        if (bytes_total) *bytes_total = 0;
        if (name_out)    *name_out    = "no device";
        return GC_E_DEVICE;
    }
    if (bytes_free)  *bytes_free  = (gc_i64)f;
    if (bytes_total) *bytes_total = (gc_i64)t;

    if (name_out) {
        static char name[256];
        static int have_name = 0;
        if (!have_name) {
            int dev = 0;
            cudaDeviceProp prop;
            if (cudaGetDevice(&dev) == cudaSuccess &&
                cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
                for (int i = 0; i < 255 && prop.name[i]; ++i) name[i] = prop.name[i];
                name[255] = '\0';
            } else {
                cudaGetLastError();
                name[0] = '\0';
            }
            have_name = 1;
        }
        *name_out = name;
    }
    return GC_OK;
}

}  /* namespace gcn */
