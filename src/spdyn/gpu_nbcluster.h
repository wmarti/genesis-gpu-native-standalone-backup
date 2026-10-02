/* gpu_nbcluster.h : cluster-pair real-space path of the native core.
 * Private to the CUDA units.  With nonbond_precision = MIXED the real-space
 * sum runs on 8-atom clusters (gpu_nbcluster.cu) instead of the cell-pair
 * walk of gpu_force.cu. */

#ifndef GPU_NBCLUSTER_H
#define GPU_NBCLUSTER_H

#include "gpu_core_native.h"

namespace gcn {

/* Is the cluster path the real-space kernel of this context? */
bool native_nbc_active(const struct gcn_device *d);

/* Exclusion list filled by the rebuild's mask clear for native_nbc_build:
 * room for at least `cap` entries, count zeroed on stream s.  All null when
 * the cluster list is not active. */
gc_status native_nbc_excl_list(struct gcn_device *d, gc_i64 cap, cudaStream_t s,
                               int4 **list, int **count, gc_i64 *room);

/* In the rebuild: the clusters of the new cells and the pair flags of the
 * first min(pcap, *np_dev) pairs, on d->stream, for the selection bounds
 * lo, hi and cmax (gcn_sel_bounds). */
gc_status native_nbc_flags(gc_context *ctx, gc_i64 pcap, const gc_i64 *np_dev,
                           float lo, float hi, double cmax);

/* After each rebuild: the clusters, the outer cluster list at the list
 * radius and its exclusion masks. */
gc_status native_nbc_build(gc_context *ctx);

/* FP32 cluster coordinates of the owned cells, on d->stream. */
gc_status native_nbc_pack_owned(gc_context *ctx);

/* One part (GCN_REAL_*) of the real-space sum on stream `s`. */
gc_status native_nbc_force(gc_context *ctx, gc_i32 want_energy,
                           cudaStream_t s, int part);

/* The box was scaled by `scale` (NPT); the next evaluation prunes again. */
gc_status native_nbc_rebox(gc_context *ctx, const double scale[3]);

/* Fixed-point overflow flag of this unit (gpu_fixed_sum.cuh). */
const unsigned int *native_nbc_overflow(const struct gcn_device *d);

void native_nbc_release(struct gcn_device *d);

}  /* namespace gcn */

#endif
