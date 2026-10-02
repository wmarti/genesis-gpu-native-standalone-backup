/*
 * gpu_rebuild.cu : the native rebuild transaction on one rank.
 *
 * Owns the device state's lifetime, the group sort, the cell runs, the pair
 * list with its selected-index runs and exclusion masks, the execution term
 * records and the list-validity guard (doc/21_GPU_Native.rst). Cell assignment
 * and the periodic offset follow sp_domain.fpp. The cell-pair stencil is
 * derived from the list radius and cell size; a geometry needing a wider
 * stencil is refused. Nothing here reads the environment.
 */

#include <climits>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/functional>
#include "gpu_core_native.h"
#include "gpu_nbcluster.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

/* The selection bit rows are 32-bit words (one ballot each). */
typedef unsigned int gc_u32;

/* Device helpers */

namespace {

/* fmix64: the hash of the device identity map below. */
__device__ __forceinline__ gc_u64 gcn_hash64(gc_u64 k)
{
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/* Exact distributed atom identity. The key is (gid, image); a slot is
 * published in three writes: atomicCAS claims key_gid, the winner stores
 * key_image, then stores val >= 0 after a __threadfence. A prober that meets
 * its own gid in a claimed slot waits for the publish before comparing
 * images. Equal (gid, image) keys count in `bad` once per duplicate. Lookups
 * (gcn_imap_find) are exact-match, so the rebuild does not depend on
 * insertion order. */
__global__ void gcn_kern_imap_clear(gc_gid *gid, gc_image *image,
                                    gc_i32 *val, gc_i64 cap)
{
    for (gc_i64 i=(gc_i64)blockIdx.x*blockDim.x+threadIdx.x; i<cap;
         i+=(gc_i64)gridDim.x*blockDim.x) {
        gid[i]=GC_GID_NONE; image[i]=0; val[i]=-1;
    }
}

__global__ void gcn_kern_imap_insert(const gc_gid *gid,
                                     const gc_image *image,
                                     gc_i64 n, gc_gid *key_gid,
                                     gc_image *key_image,
                                     gc_i32 *val, gc_i64 cap,
                                     gc_i64 *bad)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid g=gid[s]; const gc_image im=image[s];
        gc_u64 mix=gcn_hash64((gc_u64)g ^ gcn_hash64((gc_u64)im));
        gc_i64 h=(gc_i64)(mix & (gc_u64)(cap-1));
        int placed=0;
        for (gc_i64 p=0; p<cap; ++p) {
            gc_i64 i=(h+p)&(cap-1);
            const gc_gid prev = (gc_gid)atomicCAS(
                (unsigned long long *)&key_gid[i],
                (unsigned long long)GC_GID_NONE, (unsigned long long)g);
            if (prev == GC_GID_NONE) {
                key_image[i]=im;
                __threadfence();
                atomicExch(&val[i], (gc_i32)s);
                placed=1; break;
            }
            if (prev == g) {
                /* the same atom's claim: wait for its image */
                while (atomicAdd(&val[i], 0) < 0) { }
                __threadfence();
                if (((volatile gc_image *)key_image)[i] == im) {
                    atomicAdd((unsigned long long *)bad,1ull);
                    placed=1; break;
                }
            }
        }
        if (!placed) atomicAdd((unsigned long long *)bad,1ull);
    }
}

__device__ __forceinline__ gc_i32 gcn_imap_find(const gc_gid *key_gid,
                                                 const gc_image *key_image,
                                                 const gc_i32 *val,
                                                 gc_i64 cap, gc_gid g,
                                                 gc_image im)
{
    gc_u64 mix=gcn_hash64((gc_u64)g ^ gcn_hash64((gc_u64)im));
    gc_i64 h=(gc_i64)(mix & (gc_u64)(cap-1));
    for (gc_i64 p=0; p<cap; ++p) {
        gc_i64 i=(h+p)&(cap-1);
        if (key_gid[i]==g && key_image[i]==im) return val[i];
        if (key_gid[i]==GC_GID_NONE) return -1;
    }
    return -1;
}

}  /* anonymous namespace */

/* Geometry kernels */

/* GENESIS's own cell assignment, sp_domain.fpp (update_outgoing_ptl). */
__device__ __forceinline__ void gcn_cell_of_coord(double x, double y, double z,
                                                  const double *origin,
                                                  const double *box,
                                                  const double *csize,
                                                  const int *ncel,
                                                  int *icx, int *icy, int *icz)
{
    double xs = x - origin[0];
    double ys = y - origin[1];
    double zs = z - origin[2];
    xs += box[0] * 0.5 - box[0] * round(xs / box[0]);
    ys += box[1] * 0.5 - box[1] * round(ys / box[1]);
    zs += box[2] * 0.5 - box[2] * round(zs / box[2]);
    int ix = (int)(xs / csize[0]);
    int iy = (int)(ys / csize[1]);
    int iz = (int)(zs / csize[2]);
    if (ix >= ncel[0]) ix = ncel[0] - 1;
    if (iy >= ncel[1]) iy = ncel[1] - 1;
    if (iz >= ncel[2]) iz = ncel[2] - 1;
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    *icx = ix; *icy = iy; *icz = iz;
}

/* The sort key of every group: its representative's cell, then the group
 * kind, then the representative's GID.  GIDs are unique, so the order is
 * total and reproducible run to run. */
__global__ void gcn_kern_group_key(const gc_f64 *__restrict__ coord,
                                   const gc_i64 *__restrict__ goff,
                                   const gc_i32 *__restrict__ gmem,
                                   const gc_u8 *__restrict__ gkind,
                                   const gc_gid *__restrict__ ggid,
                                   gc_i32 *__restrict__ gcell,
                                   gc_i32 *__restrict__ gdest_rank,
                                   gc_i32 *__restrict__ gdest_coord,
                                   gc_u64 *__restrict__ kh,
                                   gc_u64 *__restrict__ kl,
                                   gc_i32 *__restrict__ kv,
                                   gc_i64 ngroup, gc_i64 nmember,
                                   gc_i64 nowned, gc_i64 pitch,
                                   gc_i64 *__restrict__ bad,
                                   double ox, double oy, double oz,
                                   double bx, double by, double bz,
                                   double cx, double cy, double cz,
                                   int nx, int ny, int nz,
                                   struct gcn_layout layout)
{
    double origin[3] = { ox, oy, oz };
    double box[3]    = { bx, by, bz };
    double csize[3]  = { cx, cy, cz };
    int    ncel[3]   = { nx, ny, nz };

    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 first=goff[g], last=goff[g+1];
        if(first<0 || last<=first || last>nmember ||
           (g==0 && first!=0) || (g==ngroup-1 && last!=nmember)) {
            gcell[g]=0x7fffffff;gdest_rank[g]=-1;
            gdest_coord[3*g]=gdest_coord[3*g+1]=gdest_coord[3*g+2]=-1;
            kh[g]=0xffffffffffffffffULL;kl[g]=(gc_u64)ggid[g];
            kv[g]=(gc_i32)g;
            atomicAdd((unsigned long long *)bad,1ULL);
            continue;
        }
        const gc_i32 rep = gmem[first];
        if(rep<0 || rep>=nowned) {
            gcell[g]=0x7fffffff;gdest_rank[g]=-1;
            gdest_coord[3*g]=gdest_coord[3*g+1]=gdest_coord[3*g+2]=-1;
            kh[g]=0xffffffffffffffffULL;kl[g]=(gc_u64)ggid[g];
            kv[g]=(gc_i32)g;
            atomicAdd((unsigned long long *)bad,1ULL);
            continue;
        }
        int ix, iy, iz;
        gcn_cell_of_coord(coord[rep], coord[pitch + rep],
                          coord[2 * pitch + rep],
                          origin, box, csize, ncel, &ix, &iy, &iz);
        gdest_rank[g] = gcn_rank_of_cell(layout, ix, iy, iz);
        gdest_coord[3*g] = ix;
        gdest_coord[3*g+1] = iy;
        gdest_coord[3*g+2] = iz;
        gc_i32 cell = gcn_box_index(layout, ix, iy, iz);
        if (cell < 0) {
            /* Past the seam of a split axis the box holds the cell at its
             * extended coordinate, a period out. */
            gc_i32 e[3] = { ix, iy, iz };
            for (int k = 0; k < 3; ++k)
                if (layout.halo[k] > 0) {
                    if (e[k] < layout.lo[k]) e[k] += ncel[k];
                    else if (e[k] >= layout.lo[k] + layout.dim[k]) e[k] -= ncel[k];
                }
            cell = gcn_box_index(layout, e[0], e[1], e[2]);
        }
        if (cell < 0) cell = 0x7fffffff;
        gcell[g] = cell;
        kh[g] = ((gc_u64)(unsigned int)cell << 8) | (gc_u64)gkind[g];
        kl[g] = (gc_u64)ggid[g];
        kv[g] = (gc_i32)g;
    }
}

/* Count whole migration units. A moving group goes straight to its owner,
 * which must be a neighbour: a move of more than one domain on an axis is
 * invalid. */
__global__ void gcn_kern_validate_group_owner(const gc_i32 *__restrict__ gcell,
                                               gc_i32 *__restrict__ dest_rank,
                                               const gc_i32 *__restrict__ dest_coord,
                                               const gc_i64 *__restrict__ goff,
                                               gc_i64 ngroup,
                                               struct gcn_layout layout,
                                               gc_i64 *__restrict__ count)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i32 c = gcell[g];
        gc_i32 rank = dest_rank[g];
        const gc_i64 atoms = goff[g+1] - goff[g];
        if (rank >= 0 && rank < layout.nproc && rank != layout.rank) {
            for (int k = 0; k < 3; ++k) {
                const gc_i32 n = layout.nd[k];
                const gc_i32 step = (gcn_axis_owner(layout.ncel[k], n,
                                     dest_coord[3*g+k]) - layout.ip[k] + n) % n;
                if (step != 0 && step != 1 && step != n - 1) rank = -1;
            }
            dest_rank[g] = rank;
        }
        if (rank < 0 || rank >= layout.nproc || atoms <= 0) {
            atomicAdd((unsigned long long *)(count+2), 1ULL);
        } else if (rank != layout.rank) {
            atomicAdd((unsigned long long *)(count+0), 1ULL);
            atomicAdd((unsigned long long *)(count+1),
                      (unsigned long long)atoms);
        } else if (c < 0 || c >= layout.ncell_box ||
                   !gcn_box_owned(layout, c)) {
            atomicAdd((unsigned long long *)(count+2), 1ULL);
        }
    }
}

/* Per-peer shipment layout in three steps, so no thread walks every group:
 * each group is checked and counted in parallel and flags itself as a mover;
 * the movers are listed in group order (a scan); one thread lays out the
 * segments walking only the movers, so each peer's segment is in group order. */
__global__ void gcn_kern_migration_count(
    const gc_i32 *dest_rank, const gc_i64 *goff, gc_i64 ngroup,
    gc_i64 nowned, gc_i32 self, gc_i32 nproc,
    gc_i64 *group_at, gc_i64 *atom_at, gc_i64 *totals,
    gc_i64 *peer_counts, gc_i32 *mover)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        group_at[g] = atom_at[g] = -1;
        mover[g] = 0;
        const gc_i32 p = dest_rank[g];
        if (p < 0 || p >= nproc || goff[g] < 0 || goff[g+1] > nowned ||
            goff[g+1] <= goff[g]) {
            atomicAdd((unsigned long long *)&totals[2], 1ull);
            continue;
        }
        if (p == self) continue;
        atomicAdd((unsigned long long *)&peer_counts[3*p], 1ull);
        atomicAdd((unsigned long long *)&peer_counts[3*p+1],
                  (unsigned long long)(goff[g+1] - goff[g]));
        mover[g] = 1;
    }
}

__global__ void gcn_kern_migration_list(const gc_i32 *mover,
                                        const gc_i64 *at, gc_i64 ngroup,
                                        gc_i32 *list)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x)
        if (mover[g]) list[at[g]] = (gc_i32)g;
}

__global__ void gcn_kern_migration_layout(
    gc_i32 nproc, gc_i64 *totals,
    gc_i64 *peer_counts, gc_i64 *peer_group_base, gc_i64 *peer_atom_base)
{
    if (blockIdx.x || threadIdx.x) return;
    gc_i64 groups=0, atoms=0, invalid=totals[2];
    for(gc_i32 p=0;p<nproc;++p) {
        /* Native canonical-term inventory is not established. A moving
         * peer's third count is deliberately invalid until that admission
         * exists; only the separate synthetic fixture knows it is zero. */
        if(peer_counts[3*p]>0)peer_counts[3*p+2]=-1;
        if(groups>GC_MAX_LOCAL_INDEX-peer_counts[3*p] ||
           atoms>GC_MAX_LOCAL_INDEX-peer_counts[3*p+1]) {
            ++invalid;break;
        }
        groups+=peer_counts[3*p];
        atoms+=peer_counts[3*p+1];
    }
    groups=atoms=0;
    for(gc_i32 p=0;p<nproc;++p) {
        peer_group_base[p]=groups;peer_atom_base[p]=atoms;
        groups+=peer_counts[3*p];atoms+=peer_counts[3*p+1];
    }
    totals[0]=groups; totals[1]=atoms; totals[2]=invalid;
}

/* Each mover's group and atom places in its peer's segment, in list order:
 * a warp per peer walks the list 32 movers at a time (nothing when the
 * layout refused). */
__global__ void gcn_kern_migration_places(
    const gc_i32 *__restrict__ dest_rank, const gc_i64 *__restrict__ goff,
    const gc_i32 *__restrict__ list, const gc_i64 *__restrict__ nmover,
    gc_i32 nproc, const gc_i64 *__restrict__ totals,
    const gc_i64 *__restrict__ peer_group_base,
    const gc_i64 *__restrict__ peer_atom_base,
    gc_i64 *__restrict__ group_at, gc_i64 *__restrict__ atom_at)
{
    const gc_i32 p = (gc_i32)blockIdx.x;
    const int lane = threadIdx.x;
    if (p >= nproc || totals[2]) return;
    gc_i64 gat = peer_group_base[p], aat = peer_atom_base[p];
    const gc_i64 n = *nmover;
    for (gc_i64 m0 = 0; m0 < n; m0 += 32) {
        const gc_i64 m = m0 + lane;
        gc_i64 g = 0, na = 0;
        int mine = 0;
        if (m < n) {
            g = list[m];
            mine = dest_rank[g] == p;
            if (mine) na = goff[g + 1] - goff[g];
        }
        const gc_u32 b = __ballot_sync(0xffffffffu, mine);
        gc_i64 inc = na;
        for (int o = 1; o < 32; o <<= 1) {
            const gc_i64 y = __shfl_up_sync(0xffffffffu, inc, o);
            if (lane >= o) inc += y;
        }
        if (mine) {
            group_at[g] = gat + __popc(b & ((1u << lane) - 1u));
            atom_at[g] = aat + inc - na;
        }
        gat += __popc(b);
        aat += __shfl_sync(0xffffffffu, inc, 31);
    }
}

template <class S>
__global__ void gcn_kern_migration_pack(
    const gc_i64 *goff, const gc_i32 *gmem,
    const gc_u8 *gkind, const gc_gid *ggid,
    const gc_i32 *dest_rank, const gc_i32 *dest_coord,
    const gc_i64 *group_at, const gc_i64 *atom_at,
    const gc_i64 *peer_atom_base,
    const gc_gid *gid, const gc_i32 *cls,
    const gc_f64 *charge, const gc_f64 *mass, const gc_f64 *inv_mass,
    const gc_f64 *coord, const gc_f64 *coord_ref,
    const gc_f64 *list_ref, const S *vel,
    const gc_f64 *vel_ref, const gc_f64 *vel_half,
    const gc_f64 *vel_full, const S *force,
    gc_i64 ngroup, gc_i64 nowned, gc_i64 pitch, gc_i64 epoch,
    gc_i32 self, gc_i64 group_capacity, gc_i64 atom_capacity,
    gcn_group_migration_wire *group_out,
    gcn_atom_migration_wire *atom_out, gc_i64 *bad)
{
    for (gc_i64 g=(gc_i64)blockIdx.x*blockDim.x+threadIdx.x;g<ngroup;
         g+=(gc_i64)gridDim.x*blockDim.x) {
        const gc_i64 ga=group_at[g], aa=atom_at[g];
        if (ga<0) continue;
        const gc_i64 n=goff[g+1]-goff[g];
        if (ga>=group_capacity || aa<0 || n<=0 ||
            aa>atom_capacity-n || dest_rank[g]==self || ggid[g]<=0 ||
            (gkind[g]==GC_GROUP_SINGLE && n!=1) ||
            (gkind[g]==GC_GROUP_WATER && n!=3) ||
            (gkind[g]==GC_GROUP_HGROUP &&
             (n<2 || n>GC_MAX_HGROUP_H+1)) ||
            gkind[g]>GC_GROUP_HGROUP) {
            atomicAdd((unsigned long long *)bad,1ULL); continue;
        }
        gcn_group_migration_wire &gr=group_out[ga];
        gr.schema=GCN_MIGRATION_RECORD_SCHEMA;
        gr.source_rank=self; gr.destination_rank=dest_rank[g];
        gr.kind=(gc_i32)gkind[g]; gr.epoch=epoch;
        gr.atom_offset=aa-peer_atom_base[dest_rank[g]];
        gr.atom_count=n; gr.group_gid=ggid[g];
        for(int k=0;k<3;++k)gr.destination_cell[k]=dest_coord[3*g+k];
        gr.pad0=0;
        for(gc_i64 j=0;j<n;++j) {
            const gc_i32 s=gmem[goff[g]+j];
            if(s<0 || s>=nowned) {
                atomicAdd((unsigned long long *)bad,1ULL); continue;
            }
            if(gid[s]<=0) {
                atomicAdd((unsigned long long *)bad,1ULL); continue;
            }
            int duplicate=0;
            for(gc_i64 prior=0;prior<j;++prior) {
                const gc_i32 before=gmem[goff[g]+prior];
                if(before>=0 && before<nowned && gid[before]==gid[s])
                    duplicate=1;
            }
            if(duplicate) {
                atomicAdd((unsigned long long *)bad,1ULL); continue;
            }
            gcn_atom_migration_wire &a=atom_out[aa+j];
            a.schema=GCN_MIGRATION_RECORD_SCHEMA;
            a.source_rank=self; a.destination_rank=dest_rank[g];
            a.member_ordinal=(gc_i32)j; a.epoch=epoch;
            a.gid=gid[s]; a.group_gid=ggid[g];
            for(int k=0;k<3;++k) {
                const gc_i64 at=(gc_i64)k*pitch+s;
                a.coord[k]=coord[at]; a.coord_ref[k]=coord_ref[at];
                a.list_ref[k]=list_ref[at]; a.vel[k]=vel[at];
                a.vel_ref[k]=vel_ref[at]; a.vel_half[k]=vel_half[at];
                a.vel_full[k]=vel_full[at]; a.force[k]=force[at];
            }
            a.charge=charge[s]; a.mass=mass[s];
            a.inv_mass=inv_mass[s]; a.cls=cls[s]; a.pad0=0;
        }
    }
}

__global__ void gcn_kern_resident_key(const gc_i32 *__restrict__ cell_of,
                                      gc_u64 *__restrict__ kh,
                                      gc_u64 *__restrict__ kl,
                                      gc_i32 *__restrict__ kv,
                                      gc_i64 nresident, gc_i64 ncell)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < nresident;
         s += (gc_i64)gridDim.x * blockDim.x) {
        gc_i32 c = cell_of[s];
        if (c < 0 || c >= ncell) c = 0x7fffffff;
        kh[s] = (gc_u64)(unsigned int)c;
        kl[s] = (gc_u64)s;
        kv[s] = (gc_i32)s;
    }
}

/* The term stage's traps, added to the step's sticky verdict (words 4-6,
 * gpu_core_native.h): unresolved endpoints, unplaced mask terms, and the mask
 * index, identity-map and carry-permutation traps. */
__global__ void gcn_kern_list_verdict(const gc_i64 *__restrict__ totals,
                                      const gc_i64 *__restrict__ report,
                                      const gc_i64 *__restrict__ perm_bad,
                                      gc_i64 *__restrict__ verdict)
{
    verdict[4] += totals[1];
    verdict[5] += totals[2];
    verdict[6] += report[4] + report[13] + perm_bad[0];
}

__global__ void gcn_kern_set_i64(gc_i64 *__restrict__ p, gc_i64 v) { *p = v; }

/* Each cell's atom count into a flat array (the scan's input), and the
 * longest run into *max_run (cleared by the caller). */
__global__ void gcn_kern_cell_atoms(const struct gcn_cell *__restrict__ cell,
                                    gc_i64 ncell, gc_i32 *__restrict__ out,
                                    gc_i64 *__restrict__ max_run)
{
    gc_i32 m = 0;
    for (gc_i64 c = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; c < ncell;
         c += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i32 n = cell[c].atom_count;
        out[c] = n;
        if (n > m) m = n;
    }
    for (int o = 16; o > 0; o >>= 1)
        m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0 && m > 0)
        atomicMax((unsigned long long *)max_run, (unsigned long long)m);
}

/* Count the positions whose sort key exceeds the next one's. */
__global__ void gcn_kern_key_order_check(const gc_u64 *__restrict__ kh,
                                         gc_i64 n, gc_i64 *__restrict__ bad)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i + 1 < n;
         i += (gc_i64)gridDim.x * blockDim.x)
        if (kh[i] > kh[i + 1]) atomicAdd((unsigned long long *)bad, 1ull);
}

__global__ void gcn_kern_resident_publish(const gc_u64 *__restrict__ kh,
                                          const gc_i32 *__restrict__ kv,
                                          gc_i32 *__restrict__ resident_slot,
                                          gc_i32 *__restrict__ resident_cell,
                                          gc_i64 nresident, gc_i64 ncell,
                                          gc_i64 *__restrict__ bad)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < nresident;
         i += (gc_i64)gridDim.x * blockDim.x) {
        gc_i64 c = (gc_i64)kh[i];
        if (c < 0 || c >= ncell) {
            atomicAdd((unsigned long long *)bad, 1ULL);
            c = 0;
        }
        resident_slot[i] = kv[i];
        resident_cell[i] = (gc_i32)c;
    }
}

/* Member counts in the sorted order, and then the new slot of every member
 * and the permutation the state arrays follow. */
__global__ void gcn_kern_group_counts(const gc_i32 *__restrict__ order,
                                      const gc_i64 *__restrict__ goff,
                                      gc_i64 *__restrict__ cnt,
                                      gc_i64 ngroup)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        gc_i32 o = order[g];
        cnt[g] = goff[o + 1] - goff[o];
    }
}

__global__ void gcn_kern_group_scatter(const gc_i32 *__restrict__ order,
                                       const gc_i64 *__restrict__ old_off,
                                       const gc_i32 *__restrict__ old_mem,
                                       const gc_u8 *__restrict__ old_kind,
                                       const gc_gid *__restrict__ old_gid,
                                       const gc_i32 *__restrict__ old_cell,
                                       const gc_i64 *__restrict__ new_off,
                                       gc_i32 *__restrict__ new_mem,
                                       gc_u8 *__restrict__ new_kind,
                                       gc_gid *__restrict__ new_gid,
                                       gc_i32 *__restrict__ new_cell,
                                       gc_i32 *__restrict__ perm,
                                       gc_i32 *__restrict__ cell_of,
                                       gc_i64 ngroup)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ngroup;
         g += (gc_i64)gridDim.x * blockDim.x) {
        gc_i32 o = order[g];
        gc_i64 b = old_off[o], e = old_off[o + 1];
        gc_i64 d = new_off[g];
        new_kind[g] = old_kind[o];
        new_gid[g]  = old_gid[o];
        new_cell[g] = old_cell[o];
        for (gc_i64 i = b; i < e; ++i) {
            gc_i32 slot = (gc_i32)(d + (i - b));
            new_mem[d + (i - b)] = slot;
            perm[slot]    = old_mem[i];
            cell_of[slot] = old_cell[o];
        }
    }
}

/* The group and resident sorts as a bucket sort: bucket = the key's cell (hi
 * >> sh: 8 for a group key, 0 for a resident key), anything else in bucket
 * nb; within a bucket an element's place is the number of elements before it
 * in (hi, lo, index) order, a total order independent of scheduling. */
__device__ __forceinline__ gc_i64 gcn_gbucket(gc_u64 h, gc_i64 nb, int sh)
{
    const gc_u64 c = h >> sh;
    return (h == ~0ull || c >= (gc_u64)nb) ? nb : (gc_i64)c;
}

__global__ void gcn_kern_gsort_count(const gc_u64 *__restrict__ kh, gc_i64 n,
                                     gc_i64 nb, int sh, gc_i32 *__restrict__ cnt)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < n;
         g += (gc_i64)gridDim.x * blockDim.x)
        atomicAdd(&cnt[gcn_gbucket(kh[g], nb, sh)], 1);
}

__global__ void gcn_kern_gsort_scatter(const gc_u64 *__restrict__ kh, gc_i64 n,
                                       gc_i64 nb, int sh,
                                       const gc_i64 *__restrict__ boff,
                                       gc_i32 *__restrict__ cursor,
                                       gc_i32 *__restrict__ tmp)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < n;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 b = gcn_gbucket(kh[g], nb, sh);
        tmp[boff[b] + atomicAdd(&cursor[b], 1)] = (gc_i32)g;
    }
}

__global__ void gcn_kern_gsort_rank(const gc_u64 *__restrict__ kh,
                                    const gc_u64 *__restrict__ kl,
                                    const gc_i32 *__restrict__ kv,
                                    const gc_i32 *__restrict__ tmp,
                                    const gc_i64 *__restrict__ boff,
                                    gc_i64 nbk, gc_i64 n,
                                    gc_u64 *__restrict__ oh,
                                    gc_u64 *__restrict__ ol,
                                    gc_i32 *__restrict__ ov)
{
    const int lane = threadIdx.x & 31;
    const gc_i64 nw = ((gc_i64)gridDim.x * blockDim.x) >> 5;
    for (gc_i64 b = ((gc_i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         b < nbk; b += nw) {
        const gc_i64 lo = boff[b], hi = (b + 1 < nbk) ? boff[b + 1] : n;
        for (gc_i64 i = lo + lane; i < hi; i += 32) {
            const gc_i32 gi = tmp[i];
            const gc_u64 hi_i = kh[gi], lo_i = kl[gi];
            gc_i64 r = 0;
            for (gc_i64 j = lo; j < hi; ++j) {
                const gc_i32 gj = tmp[j];
                const gc_u64 hj = kh[gj], lj = kl[gj];
                r += (hj < hi_i) || (hj == hi_i && (lj < lo_i ||
                                                    (lj == lo_i && gj < gi)));
            }
            oh[lo + r] = hi_i; ol[lo + r] = lo_i; ov[lo + r] = kv[gi];
        }
    }
}

/* The whole resident state in one gather into the second set: six
 * (3, pitch) vector arrays and the five per-atom arrays, perm read once per
 * owned slot (new[s] = old[perm[s]]); the ghosts keep their slots. */
/* v[2] is the velocity (gcn_vf), of element type S */
struct gcn_perm_set {
    void   *v[6];
    gc_f64 *charge, *mass, *inv_mass;
    gc_i32 *cls;
    gc_gid *gid;
};

template <class S>
__global__ void gcn_kern_permute_all(struct gcn_perm_set a,
                                     struct gcn_perm_set b,
                                     const gc_i32 *__restrict__ perm,
                                     gc_i64 n, gc_i64 nres, gc_i64 pitch)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < nres;
         s += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 o = s < n ? (gc_i64)perm[s] : s;
        for (int k = 0; k < 6; ++k)
            for (int c = 0; c < 3; ++c)
                if (k == 2)
                    ((S *)b.v[k])[c * pitch + s] =
                        ((const S *)a.v[k])[c * pitch + o];
                else
                    ((gc_f64 *)b.v[k])[c * pitch + s] =
                        ((const gc_f64 *)a.v[k])[c * pitch + o];
        b.charge[s]   = a.charge[o];
        b.mass[s]     = a.mass[o];
        b.inv_mass[s] = a.inv_mass[o];
        b.cls[s] = a.cls[o];
        b.gid[s] = a.gid[o];
    }
}

/* The cell runs, from the sorted groups: each cell's first slot, its slot
 * count, and the same for groups. */
__global__ void gcn_kern_cell_runs(const gc_i32 *__restrict__ cell_of,
                                   const gc_i32 *__restrict__ gcell,
                                   struct gcn_cell *__restrict__ cell,
                                   gc_i64 natom, gc_i64 ngroup, gc_i64 ncell,
                                   struct gcn_layout layout)
{
    for (gc_i64 c = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; c < ncell;
         c += (gc_i64)gridDim.x * blockDim.x) {
        cell[c].atom_begin  = 0;
        cell[c].atom_count  = 0;
        cell[c].group_begin = 0;
        cell[c].group_count = 0;
        gcn_box_coords(layout, (gc_i32)c,
                       &cell[c].gx, &cell[c].gy, &cell[c].gz);
        cell[c].pad0 = 0;
    }
    (void)cell_of; (void)gcell; (void)natom; (void)ngroup;
}

/* One thread per element; a boundary between two different keys marks the
 * start of a run and the element before the next boundary its end. The two
 * ends are written by separate kernels, each deriving its own value; the
 * count is the difference of the two positions. */
__global__ void gcn_kern_cell_bounds(const gc_i32 *__restrict__ key,
                                     gc_i64 n, struct gcn_cell *cell,
                                     int which, gc_i32 *__restrict__ endpos)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        gc_i32 k = key[i];
        if (i == 0 || key[i - 1] != k) {
            if (which == 0) cell[k].atom_begin  = (gc_i32)i;
            else            cell[k].group_begin = (gc_i32)i;
        }
        if (i == n - 1 || key[i + 1] != k) endpos[k] = (gc_i32)(i + 1);
    }
}

__global__ void gcn_kern_cell_counts(struct gcn_cell *cell,
                                     const gc_i32 *__restrict__ endpos,
                                     gc_i64 ncell, int which)
{
    for (gc_i64 c = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; c < ncell;
         c += (gc_i64)gridDim.x * blockDim.x) {
        gc_i32 e = endpos[c];
        if (which == 0)
            cell[c].atom_count  = (e > 0) ? e - cell[c].atom_begin  : 0;
        else
            cell[c].group_count = (e > 0) ? e - cell[c].group_begin : 0;
    }
}

/* Pair enumeration */

/* The stencil, as the rebuild derives it, packed so the kernels agree. */
struct gcn_stencil {
    int nh[3];        /* half width in cells, per axis                   */
    int span[3];      /* 2*nh + 1                                        */
    int size;         /* span[0]*span[1]*span[2]                         */
    int ncel[3];      /* global cell counts                              */
    struct gcn_layout layout; /* this rank's compact owned+halo cell box       */
    double box[3];
    double list2;     /* pairlistdist^2                                  */
};

__device__ __forceinline__ void gcn_offset_of(const struct gcn_stencil &st,
                                              int o, int *dx, int *dy, int *dz)
{
    int t = o;
    *dx = t % st.span[0] - st.nh[0]; t /= st.span[0];
    *dy = t % st.span[1] - st.nh[1]; t /= st.span[1];
    *dz = t - st.nh[2];
}

__device__ __forceinline__ int gcn_ordinal_of(const struct gcn_stencil &st,
                                              int dx, int dy, int dz)
{
    return (dx + st.nh[0])
         + st.span[0] * ((dy + st.nh[1])
         + st.span[1] * (dz + st.nh[2]));
}

/* Candidate enumeration. A pair is kept once, from the endpoint
 * gcn_pair_home_is picks. The periodic offset is the whole-box shift of the
 * minimum image (the three-candidate form of sp_domain.fpp) and the virial
 * flag is GENESIS's virial_check. */
__device__ __forceinline__ gc_i64 gcn_global_cell_key(struct gcn_stencil st,
                                                               int ex, int ey, int ez,
                                                               int *wx, int *wy, int *wz)
{
    int x = gcn_wrap_axis(ex, st.ncel[0], wx);
    int y = gcn_wrap_axis(ey, st.ncel[1], wy);
    int z = gcn_wrap_axis(ez, st.ncel[2], wz);
    return ((gc_i64)z * st.ncel[1] + y) * st.ncel[0] + x;
}

/* The half-list rule: does the cell keyed `self` enumerate its pair with the
 * distinct cell keyed `other`? It reads the two wrapped global keys only, so
 * every rank decides alike and exactly one endpoint's owner evaluates the
 * pair. A hash bit of the unordered pair picks the lower- or higher-keyed
 * end, which balances cross-rank pairs (a plain lower-key rule overloads the
 * low-corner rank). */
__device__ __forceinline__ bool gcn_pair_home_is(gc_i64 self, gc_i64 other)
{
    const gc_u64 lo = (gc_u64)(self < other ? self : other);
    const gc_u64 hi = (gc_u64)(self < other ? other : self);
    gc_u64 h = (lo << 32) | hi;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (self < other) != (bool)(h & 1);
}

/* Which endpoint cell is the pair's home (the cell the enumerator enumerated
 * from, fixed by gcn_pair_home_is), and the displacement from it to the other
 * cell in the enumerator's frame (gcn_box_cell_disp's: on an axis this rank
 * owns whole, the raw difference to a wrapped ghost equals the offset only
 * modulo the period). Returns the home slot, or -1 when this rank does not
 * own the real-space copy of the unordered cell pair; its owner clears that
 * mask, so it is not a failure here. */
__device__ __forceinline__ gc_i32 gcn_mask_home_other(
    const struct gcn_stencil &st, const struct gcn_cell *cell,
    gc_i32 ca, gc_i32 cb, gc_i32 *other, gc_i32 *d)
{
    int w[3];
    const gc_i64 ka = gcn_global_cell_key(st, cell[ca].gx, cell[ca].gy,
                                          cell[ca].gz, &w[0], &w[1], &w[2]);
    const gc_i64 kb = gcn_global_cell_key(st, cell[cb].gx, cell[cb].gy,
                                          cell[cb].gz, &w[0], &w[1], &w[2]);
    gc_i32 home;
    if (ca == cb && gcn_box_owned(st.layout, ca)) {
        home = ca; *other = cb;
    } else if (ka != kb && gcn_pair_home_is(ka, kb) &&
               gcn_box_owned(st.layout, ca)) {
        home = ca; *other = cb;
    } else if (ka != kb && gcn_pair_home_is(kb, ka) &&
               gcn_box_owned(st.layout, cb)) {
        home = cb; *other = ca;
    } else {
        return -1;
    }
    gcn_box_cell_disp(st.layout, cell[home].gx, cell[home].gy, cell[home].gz,
                      cell[*other].gx, cell[*other].gy, cell[*other].gz, d);
    return home;
}

__global__ void gcn_kern_pair_flag(const struct gcn_cell *__restrict__ cell,
                                   gc_i32 *__restrict__ keep,
                                   gc_i64 ncell, struct gcn_stencil st)
{
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
         t < ncell * st.size; t += (gc_i64)gridDim.x * blockDim.x) {
        gc_i64 ci = t / st.size;
        int o = (int)(t - ci * st.size);
        if (!gcn_box_owned(st.layout, (gc_i32)ci)) { keep[t] = 0; continue; }
        int dx, dy, dz;
        gcn_offset_of(st, o, &dx, &dy, &dz);
        const int exj = cell[ci].gx + dx;
        const int eyj = cell[ci].gy + dy;
        const int ezj = cell[ci].gz + dz;
        const gc_i32 cj = gcn_box_index(st.layout, exj, eyj, ezj);
        if (cj < 0) { keep[t] = 0; continue; }
        int wix, wiy, wiz, wjx, wjy, wjz;
        const gc_i64 ki = gcn_global_cell_key(st, cell[ci].gx, cell[ci].gy,
                                              cell[ci].gz, &wix, &wiy, &wiz);
        const gc_i64 kj = gcn_global_cell_key(st, exj, eyj, ezj,
                                              &wjx, &wjy, &wjz);
        (void)wix; (void)wiy; (void)wiz;
        const int take = (kj != ki && gcn_pair_home_is(ki, kj)) ||
                         (kj == ki && dx == 0 && dy == 0 && dz == 0);
        keep[t] = take ? 1 : 0;
    }
}

__global__ void gcn_kern_pair_fill(const struct gcn_cell *__restrict__ cell,
                                   const gc_i32 *__restrict__ keep,
                                   const gc_i64 *__restrict__ slot,
                                   gc_i32 *__restrict__ pair_at,
                                   struct gcn_pair *__restrict__ pair,
                                   gc_i64 cap, gc_i64 ncell,
                                   struct gcn_stencil st)
{
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
         t < ncell * st.size; t += (gc_i64)gridDim.x * blockDim.x) {
        gc_i64 ci = t / st.size;
        int o = (int)(t - ci * st.size);
        if (!keep[t]) { pair_at[t] = -1; continue; }
        int dx, dy, dz;
        gcn_offset_of(st, o, &dx, &dy, &dz);
        const int exj = cell[ci].gx + dx;
        const int eyj = cell[ci].gy + dy;
        const int ezj = cell[ci].gz + dz;
        const gc_i32 cj = gcn_box_index(st.layout, exj, eyj, ezj);
        if (cj < 0) { pair_at[t] = -1; continue; }
        int wix, wiy, wiz, wx, wy, wz;
        (void)gcn_global_cell_key(st, cell[ci].gx, cell[ci].gy, cell[ci].gz,
                                  &wix, &wiy, &wiz);
        (void)gcn_global_cell_key(st, exj, eyj, ezj, &wx, &wy, &wz);

        gc_i64 p = slot[t];
        pair_at[t] = (gc_i32)p;
        if (p >= cap) continue;   /* outgrown: the build counts again */
        pair[p].ci = (gc_i32)ci;
        pair[p].cj = cj;
        pair[p].move[0] = -(double)(wx - wix) * st.box[0];
        pair[p].move[1] = -(double)(wy - wiy) * st.box[1];
        pair[p].move[2] = -(double)(wz - wiz) * st.box[2];
        pair[p].virial = ((wx-wix) || (wy-wiy) || (wz-wiz)) ? 1 : 0;
        pair[p].self_pair = (cj == ci && dx == 0 && dy == 0 && dz == 0) ? 1 : 0;
        pair[p].ix_off = 0; pair[p].ix_n = 0;
        pair[p].iy_off = 0; pair[p].iy_n = 0;
        pair[p].mask_off = -1;
    }
}

/* Count, then fill, the selected-index runs. An i atom is selected when some
 * j atom of the partner cell is within the list radius of it, and vice versa;
 * a self pair selects every atom of its cell.
 *
 * The count pass decides each atom's selection once and keeps one bit per
 * atom per side (`hit`, `words` 32-bit words per side; pair p's i side at
 * hit[2*p*words], j side at hit[(2*p+1)*words]). The fill pass places each
 * selected atom at the place sel_count2 computed, or, for unstaged pairs, at
 * the number of selected atoms before it, read off those bits with popcounts.
 *
 * The test is FP64: (c_a + move) - c_b squared against list2. An FP32 copy
 * decides it first only where it cannot differ: lo/hi (gcn_sel_bounds)
 * bracket list2 by more than the worst FP32 error, so d2f < lo proves d2 <
 * list2 and d2f > hi proves the opposite; anything between, and any NaN,
 * takes the FP64 test. The selected set is bit-identical to the pure FP64
 * pass. */
#define GCN_SEL_CHUNK GCN_BLOCK

__device__ __forceinline__ int gcn_sel_side(
    const gc_f64 *__restrict__ coord, const gc_i32 *__restrict__ resident_slot,
    const struct gcn_cell &cs, const struct gcn_cell &cp,
    double mx, double my, double mz, gc_i64 pitch, double list2,
    float lo, float hi, double cmax, gc_u32 *__restrict__ bits,
    float *s_x, float *s_y, float *s_z, int *s_slot)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    int count = 0;
    for (int a0 = 0; a0 < cs.atom_count; a0 += GCN_SEL_CHUNK) {
        const int a = a0 + tid;
        double x = 0.0, y = 0.0, z = 0.0;
        float xf = 0.f, yf = 0.f, zf = 0.f;
        int hit = 0;
        if (a < cs.atom_count) {
            const int sa = resident_slot[cs.atom_begin + a];
            x = coord[sa] + mx;
            y = coord[pitch + sa] + my;
            z = coord[2 * pitch + sa] + mz;
            xf = (float)x; yf = (float)y; zf = (float)z;
            /* outside the bound the error analysis assumes: a NaN copy
               sends every test of this atom to the FP64 path */
            if (!(fabs(x) <= cmax && fabs(y) <= cmax && fabs(z) <= cmax))
                xf = __int_as_float(0x7fc00000);
        }
        for (int b0 = 0; b0 < cp.atom_count; b0 += GCN_SEL_CHUNK) {
            __syncthreads();
            const int b = b0 + tid;
            if (b < cp.atom_count) {
                const int sb = resident_slot[cp.atom_begin + b];
                const double bx = coord[sb], by = coord[pitch + sb],
                             bz = coord[2 * pitch + sb];
                s_x[tid] = (fabs(bx) <= cmax && fabs(by) <= cmax &&
                            fabs(bz) <= cmax) ? (float)bx
                                              : __int_as_float(0x7fc00000);
                s_y[tid] = (float)by;
                s_z[tid] = (float)bz;
                s_slot[tid] = sb;
            }
            __syncthreads();
            const int nb = min(GCN_SEL_CHUNK, cp.atom_count - b0);
            if (a < cs.atom_count && !hit) {
                for (int k = 0; k < nb; ++k) {
                    const float dxf = xf - s_x[k];
                    const float dyf = yf - s_y[k];
                    const float dzf = zf - s_z[k];
                    const float d2f = dxf * dxf + dyf * dyf + dzf * dzf;
                    if (d2f < lo) { hit = 1; break; }
                    if (d2f > hi) continue;
                    const int sb = s_slot[k];
                    const double dx = x - coord[sb];
                    const double dy = y - coord[pitch + sb];
                    const double dz = z - coord[2 * pitch + sb];
                    if (dx*dx + dy*dy + dz*dz < list2) { hit = 1; break; }
                }
            }
        }
        const gc_u32 w = __ballot_sync(0xffffffffu, hit);
        if (lane == 0 && a0 + warp * 32 < cs.atom_count) {
            bits[(a0 >> 5) + warp] = w;
            count += __popc(w);
        }
    }
    return count;   /* meaningful in each warp's lane 0 */
}

/* Every cell-run entry's FP32 view for the selection, in cell-run order
 * (entry k is slot resident_slot[k]): (float)coord per component, x NaN when
 * any |coord| exceeds c1 = cmax / 2 (that atom is decided in FP64). A moved
 * view is view + (float)move; with |coord|, |move| <= c1 each view is within
 * 3u cmax of the FP64 value (gcn_sel_bounds). */
__global__ void gcn_kern_atom_view(const gc_f64 *__restrict__ coord,
                                   const gc_i32 *__restrict__ resident_slot,
                                   gc_i64 n, gc_i64 pitch, double c1,
                                   float4 *__restrict__ av)
{
    for (gc_i64 k = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; k < n;
         k += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 i = resident_slot[k];
        const double x = coord[i], y = coord[pitch + i], z = coord[2 * pitch + i];
        const int ok = fabs(x) <= c1 && fabs(y) <= c1 && fabs(z) <= c1;
        av[k] = make_float4(ok ? (float)x : __int_as_float(0x7fc00000),
                            (float)y, (float)z, 0.f);
    }
}

/* The FP32 box of every cell's unmoved views, and `nan` set when any of
 * them is NaN (no box argument is then made for the cell).  One warp per
 * cell. */
struct gcn_cbox { float lo[3]; float hi[3]; int nan; int pad; };

__global__ void gcn_kern_cell_box(const float4 *__restrict__ av,
                                  const struct gcn_cell *__restrict__ cell,
                                  gc_i64 ncell, struct gcn_cbox *__restrict__ box)
{
    /* grid-stride: grid_for caps the grid */
    const int lane = threadIdx.x & 31;
    const gc_i64 nw = ((gc_i64)gridDim.x * blockDim.x) >> 5;
    for (gc_i64 c = ((gc_i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         c < ncell; c += nw) {
    const struct gcn_cell cc = cell[c];
    float lo[3] = { INFINITY, INFINITY, INFINITY };
    float hi[3] = { -INFINITY, -INFINITY, -INFINITY };
    int nan = 0;
    for (int a = lane; a < cc.atom_count; a += 32) {
        const float4 v = av[cc.atom_begin + a];
        if (v.x != v.x) nan = 1;
        lo[0] = fminf(lo[0], v.x); hi[0] = fmaxf(hi[0], v.x);
        lo[1] = fminf(lo[1], v.y); hi[1] = fmaxf(hi[1], v.y);
        lo[2] = fminf(lo[2], v.z); hi[2] = fmaxf(hi[2], v.z);
    }
    for (int o = 16; o > 0; o >>= 1) {
        for (int k = 0; k < 3; ++k) {
            lo[k] = fminf(lo[k], __shfl_xor_sync(0xffffffffu, lo[k], o));
            hi[k] = fmaxf(hi[k], __shfl_xor_sync(0xffffffffu, hi[k], o));
        }
        nan |= __shfl_xor_sync(0xffffffffu, nan, o);
    }
    if (lane == 0) {
        struct gcn_cbox b;
        for (int k = 0; k < 3; ++k) { b.lo[k] = lo[k]; b.hi[k] = hi[k]; }
        b.nan = nan; b.pad = 0;
        box[c] = b;
    }
    }
}

/* The selection count with both sides of a pair decided in one pass over
 * their distance matrix, plus a box skip, for pairs whose cells both fit the
 * staging (GCN_SEL_STAGE atoms; others take gcn_sel_side). Decisions are
 * gcn_sel_side's: one FP32 d2f serves both sides, and where it does not
 * decide each side takes its own FP64 form. The box skip applies when the
 * partners' FP32 box lies beyond `hibox` (hi plus a few ulps) and no copy
 * involved is NaN. Bits and counts equal gcn_sel_side's bit for bit. */
#define GCN_SEL_STAGE 128
/* the place histogram is four counts per lane */
static_assert(GCN_SEL_STAGE == 128, "the place histogram assumes 128 bins");

__device__ __forceinline__ float gcn_box_d2(float x, float y, float z,
                                            float lx, float ly, float lz,
                                            float hx, float hy, float hz)
{
    const float ex = fmaxf(0.f, fmaxf(lx - x, x - hx));
    const float ey = fmaxf(0.f, fmaxf(ly - y, y - hy));
    const float ez = fmaxf(0.f, fmaxf(lz - z, z - hz));
    return ex * ex + ey * ey + ez * ez;
}

/* A warp's staging for one pair: the active atoms' FP32 views (the i side
 * moved) and their run positions, and every atom's partner count.  After
 * the pass the views' space holds the selected atoms' counts. */
struct gcn_sel_stage {
    float4 xy[2][GCN_SEL_STAGE / 2];   /* atoms 2t, 2t+1: x, x, y, y */
    float2 z[2][GCN_SEL_STAGE / 2];    /* atoms 2t, 2t+1: z, z       */
    unsigned char cnt[2][GCN_SEL_STAGE];      /* at most GCN_SEL_STAGE */
    unsigned char act[2][GCN_SEL_STAGE];
};

__device__ __forceinline__ void gcn_sel_put(struct gcn_sel_stage &sg, int side,
                                            int k, float x, float y, float z)
{
    float *const xy = (float *)&sg.xy[side][k >> 1];
    xy[k & 1] = x;
    xy[2 + (k & 1)] = y;
    ((float *)&sg.z[side][0])[k] = z;
}

/* Two FP32 lanes at once: one f32x2 instruction where the target has it,
 * else the two single operations it rounds exactly as. */
__device__ __forceinline__ float2 gcn_f2_sub(float2 a, float2 b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    float2 r;
    asm("{ .reg .b64 a, b, r;\n\t"
        "mov.b64 a, {%2, %3};\n\tmov.b64 b, {%4, %5};\n\t"
        "sub.rn.f32x2 r, a, b;\n\tmov.b64 {%0, %1}, r; }"
        : "=f"(r.x), "=f"(r.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return r;
#else
    return make_float2(__fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y));
#endif
}
__device__ __forceinline__ float2 gcn_f2_mul(float2 a, float2 b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    float2 r;
    asm("{ .reg .b64 a, b, r;\n\t"
        "mov.b64 a, {%2, %3};\n\tmov.b64 b, {%4, %5};\n\t"
        "mul.rn.f32x2 r, a, b;\n\tmov.b64 {%0, %1}, r; }"
        : "=f"(r.x), "=f"(r.y) : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
    return r;
#else
    return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}
__device__ __forceinline__ float2 gcn_f2_fma(float2 a, float2 b, float2 c)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    float2 r;
    asm("{ .reg .b64 a, b, c, r;\n\t"
        "mov.b64 a, {%2, %3};\n\tmov.b64 b, {%4, %5};\n\t"
        "mov.b64 c, {%6, %7};\n\t"
        "fma.rn.f32x2 r, a, b, c;\n\tmov.b64 {%0, %1}, r; }"
        : "=f"(r.x), "=f"(r.y)
        : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
    return r;
#else
    return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
#endif
}

/* The transpose of a 32 x 32 bit matrix held a row per lane (bit c of lane
 * r is entry (r, c)): lane c gets column c.  Each level swaps the
 * off-diagonal blocks of width j. */
__device__ __forceinline__ gc_u32 gcn_bits_transpose(gc_u32 x, int lane)
{
    const gc_u32 lo[5] = { 0x0000ffffu, 0x00ff00ffu, 0x0f0f0f0fu,
                           0x33333333u, 0x55555555u };
#pragma unroll
    for (int l = 0, j = 16; l < 5; ++l, j >>= 1) {
        const gc_u32 t = __shfl_xor_sync(0xffffffffu, x, j);
        x = (lane & j) ? (x & ~lo[l]) | ((t & ~lo[l]) >> j)
                       : (x & lo[l]) | ((t & lo[l]) << j);
    }
    return x;
}

/* The hit (d2f < lo) and far (d2f > hi) bits of one lane's view against
 * partners kb .. kb + 2 NP - 1 of side J, partner u at bit u: the signs of
 * d2f - lo and hi - d2f, shifted in from the last partner down (exact for
 * every non-NaN d2f). */
template <int NP>
__device__ __forceinline__ void gcn_sel_block(const struct gcn_sel_stage &sg,
                                              int J, int kb, float2 qx2,
                                              float2 qy2, float2 qz2,
                                              float2 lo2, float2 hi2,
                                              gc_u32 &m, gc_u32 &far)
{
#pragma unroll
    for (int t = NP - 1; t >= 0; --t) {
        const float4 kxy = sg.xy[J][(kb >> 1) + t];
        const float2 kz = sg.z[J][(kb >> 1) + t];
        const float2 dx = gcn_f2_sub(qx2, make_float2(kxy.x, kxy.y));
        const float2 dy = gcn_f2_sub(qy2, make_float2(kxy.z, kxy.w));
        const float2 dz = gcn_f2_sub(qz2, kz);
        const float2 d2 = gcn_f2_fma(dz, dz,
                                     gcn_f2_fma(dy, dy, gcn_f2_mul(dx, dx)));
        const float2 sl = gcn_f2_sub(d2, lo2);
        const float2 sh = gcn_f2_sub(hi2, d2);
        m   = __funnelshift_l(__float_as_uint(sl.y), m, 1);
        far = __funnelshift_l(__float_as_uint(sh.y), far, 1);
        m   = __funnelshift_l(__float_as_uint(sl.x), m, 1);
        far = __funnelshift_l(__float_as_uint(sh.x), far, 1);
    }
}

#define GCN_SEL_GRAB 8

/* One staged pair, decided by one warp: stage the views of both cells (atoms
 * within hibox of the other cell's FP32 box); one pass over the distance
 * matrix decides both sides (d2f < lo: both atoms; d2f > hi: neither; the
 * band and NaN views take each side's FP64 form); then the selection bits and
 * each selected atom's place in its run: descending partner count, ties by
 * run position (the order of GENESIS's kern_build_pairlist). */
__device__ __forceinline__ void gcn_sel_pair_warp(
    gc_i64 p, const struct gcn_cell &ci, const struct gcn_cell &cj,
    double mx, double my, double mz,
    const gc_f64 *__restrict__ coord, const float4 *__restrict__ av,
    const gc_i32 *__restrict__ resident_slot,
    const struct gcn_cbox &bxi, const struct gcn_cbox &bxj,
    gc_u32 *__restrict__ hit, gc_i64 words, gc_u8 *__restrict__ place,
    gc_i64 pitch, double list2, float lo, float hi, float hibox,
    struct gcn_sel_stage &sg, int *n_out)
{
    const int lane = threadIdx.x & 31;
    const gc_u32 lt = (1u << lane) - 1u;
    const int ni = ci.atom_count, nj = cj.atom_count;
    const float mxf = (float)mx, myf = (float)my, mzf = (float)mz;
    const float mlx = bxi.lo[0] + mxf, mly = bxi.lo[1] + myf,
                mlz = bxi.lo[2] + mzf;
    const float mhx = bxi.hi[0] + mxf, mhy = bxi.hi[1] + myf,
                mhz = bxi.hi[2] + mzf;
    int na = 0, nb = 0;
    for (int t0 = 0; t0 < ni + nj; t0 += 32) {
        const int t = t0 + lane;
        const int side = t < ni ? 0 : 1;
        const int a = side ? t - ni : t;
        int act = 0;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (t < ni + nj) {
            v = av[(side ? cj.atom_begin : ci.atom_begin) + a];
            if (side == 0) {
                v.x = v.x + mxf; v.y = v.y + myf; v.z = v.z + mzf;
                act = v.x != v.x || bxj.nan ||
                      !(gcn_box_d2(v.x, v.y, v.z, bxj.lo[0], bxj.lo[1],
                                   bxj.lo[2], bxj.hi[0], bxj.hi[1],
                                   bxj.hi[2]) > hibox);
            } else {
                act = v.x != v.x || bxi.nan ||
                      !(gcn_box_d2(v.x, v.y, v.z, mlx, mly, mlz,
                                   mhx, mhy, mhz) > hibox);
            }
            sg.cnt[side][a] = 0;
        }
        const gc_u32 b0 = __ballot_sync(0xffffffffu, act && side == 0);
        const gc_u32 b1 = __ballot_sync(0xffffffffu, act && side == 1);
        if (act) {
            const int k = side ? nb + __popc(b1 & lt) : na + __popc(b0 & lt);
            gcn_sel_put(sg, side, k, v.x, v.y, v.z);
            sg.act[side][k] = (unsigned char)a;
        }
        na += __popc(b0);
        nb += __popc(b1);
    }
    /* each side padded to a multiple of 32 with a view no partner reaches:
       its d2f overflows to +inf (not NaN) against every finite view */
    if (na + lane < ((na + 31) & ~31))
        gcn_sel_put(sg, 0, na + lane, 1.0e30f, 1.0e30f, 1.0e30f);
    if (nb + lane < ((nb + 31) & ~31))
        gcn_sel_put(sg, 1, nb + lane, 1.0e30f, 1.0e30f, 1.0e30f);
    __syncwarp();
    if (na > 0 && nb > 0) {
        const int L = ((nb + 31) >> 5) * ((na + 15) >> 4) <
                      ((na + 31) >> 5) * ((nb + 15) >> 4);
        const int J = 1 - L;
        const int nl = L ? nb : na, nk = L ? na : nb;
        const int ai_l = L ? cj.atom_begin : ci.atom_begin;
        const int ai_k = L ? ci.atom_begin : cj.atom_begin;
        for (int c0 = 0; c0 < nl; c0 += 32) {
            const int idx = c0 + lane;
            const int ok = idx < nl;
            float qx = INFINITY, qy = 0.f, qz = 0.f;
            if (ok) {
                const float *const xy = (const float *)&sg.xy[L][idx >> 1];
                qx = xy[idx & 1]; qy = xy[2 + (idx & 1)];
                qz = ((const float *)&sg.z[L][0])[idx];
            }
            const int lnan = qx != qx;
            const float2 qx2 = make_float2(qx, qx), qy2 = make_float2(qy, qy),
                         qz2 = make_float2(qz, qz);
            const float2 lo2 = make_float2(lo, lo), hi2 = make_float2(hi, hi);
            int cl = 0;
            for (int kb = 0; kb < nk; kb += 32) {
                const int nu = min(32, nk - kb);
                /* the lane's hits (d < lo) and far entries (d > hi) against
                   the next 32 partners, a bit each: the sign of d2f - lo and
                   of hi - d2f, shifted in from partner 31 down (exact for
                   every non-NaN d2f; the NaN views are marked below) */
                gc_u32 m = 0u, far = 0u;
                if (nu > 16)
                    gcn_sel_block<16>(sg, J, kb, qx2, qy2, qz2, lo2, hi2, m, far);
                else
                    gcn_sel_block<8>(sg, J, kb, qx2, qy2, qz2, lo2, hi2, m, far);
                /* undecided: the band and every NaN view, this lane's or a
                   partner's */
                const float kxn = ((const float *)&sg.xy[J][(kb + lane) >> 1])[lane & 1];
                const gc_u32 knan = __ballot_sync(0xffffffffu, kxn != kxn);
                gc_u32 bm = lnan ? 0xffffffffu : (~(m | far) | knan);
                m &= ~bm;
                if (!ok) bm = 0u;
                if (nu < 32) bm &= (1u << nu) - 1u;
                gc_u32 mk = m;
                /* the undecided entries: each side's own FP64 form */
                if (__any_sync(0xffffffffu, bm)) {
                    const int al = sg.act[L][ok ? idx : 0];
                    for (; bm; bm &= bm - 1u) {
                        const int u = __ffs(bm) - 1;
                        const int ak = sg.act[J][kb + u];
                        const int sa = resident_slot[L ? ai_k + ak : ai_l + al];
                        const int sb = resident_slot[L ? ai_l + al : ai_k + ak];
                        const double xa = coord[sa], ya = coord[pitch + sa],
                                     za = coord[2 * pitch + sa];
                        const double xb = coord[sb], yb = coord[pitch + sb],
                                     zb = coord[2 * pitch + sb];
                        int h0, h1;
                        {
                            const double x = xa + mx, y = ya + my, z = za + mz;
                            const double dx = x - xb, dy = y - yb, dz = z - zb;
                            h0 = dx*dx + dy*dy + dz*dz < list2;
                        }
                        {
                            const double x = xb + (-mx), y = yb + (-my),
                                         z = zb + (-mz);
                            const double dx = x - xa, dy = y - ya, dz = z - za;
                            h1 = dx*dx + dy*dy + dz*dz < list2;
                        }
                        if (L ? h1 : h0) m |= 1u << u;
                        if (L ? h0 : h1) mk |= 1u << u;
                    }
                }
                cl += __popc(m);
                const gc_u32 col = gcn_bits_transpose(mk, lane);
                if (lane < nu && col) sg.cnt[J][sg.act[J][kb + lane]] += __popc(col);
            }
            if (ok && cl) sg.cnt[L][sg.act[L][idx]] += cl;
            __syncwarp();
        }
    }
    __syncwarp();
    /* the bits, and each selected atom's place: the atoms with a larger
       count, from a histogram of the counts (in the views' space), plus those
       before it with the same count, counted in run order and kept in act */
    int *const hg = (int *)&sg.xy[0][0];
    int ns[2] = { 0, 0 };
    for (int side = 0; side < 2; ++side) {
        const int n = side ? nj : ni;
        int *const h = hg + side * GCN_SEL_STAGE;
        reinterpret_cast<int4 *>(h)[lane] = make_int4(0, 0, 0, 0);
        __syncwarp();
        for (int c0 = 0; c0 < n; c0 += 32) {
            const int a = c0 + lane;
            const int c = a < n ? sg.cnt[side][a] : 0;
            const gc_u32 w = __ballot_sync(0xffffffffu, c > 0);
            if (lane == 0) hit[(2 * p + side) * words + (c0 >> 5)] = w;
            ns[side] += __popc(w);
            const gc_u32 m = __match_any_sync(0xffffffffu, c);
            const int lead = __ffs(m) - 1;
            const int before = (c > 0 && lane == lead) ? h[c - 1] : 0;
            const int e = __shfl_sync(0xffffffffu, before, lead) + __popc(m & lt);
            if (c > 0 && lane == lead) h[c - 1] = before + __popc(m);
            if (c > 0) sg.act[side][a] = (unsigned char)e;
            __syncwarp();
        }
        const int4 q = reinterpret_cast<const int4 *>(h)[lane];
        const int loc = q.x + q.y + q.z + q.w;
        int above = loc;
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_down_sync(0xffffffffu, above, o);
            if (lane + o < 32) above += t;
        }
        above -= loc;
        __syncwarp();
        reinterpret_cast<int4 *>(h)[lane] =
            make_int4(above + q.y + q.z + q.w, above + q.z + q.w, above + q.w, above);
        __syncwarp();
        for (int c0 = 0; c0 < n; c0 += 32) {
            const int a = c0 + lane;
            const int c = a < n ? sg.cnt[side][a] : 0;
            if (c > 0)
                place[(2 * p + side) * 32 * words + a] =
                    (gc_u8)(h[c - 1] + sg.act[side][a]);
        }
        __syncwarp();
    }
    n_out[0] = ns[0];
    n_out[1] = ns[1];
}

/* The selection count: a warp per pair, the warps of a resident grid taking
 * GCN_SEL_GRAB pairs at a time from queue[0] (zeroed by the caller). A self
 * pair selects every atom; a pair with a run longer than the bits is counted
 * in `overflow`; a pair with a cell over GCN_SEL_STAGE atoms is listed in
 * big[] (queue[1] of them) for gcn_kern_sel_count_big. */
__global__ void gcn_kern_sel_count2(const gc_f64 *__restrict__ coord,
                                    const float4 *__restrict__ av,
                                    const struct gcn_cell *__restrict__ cell,
                                    const gc_i32 *__restrict__ resident_slot,
                                    const struct gcn_pair *__restrict__ pair,
                                    gc_i32 *__restrict__ cnt_i,
                                    gc_i32 *__restrict__ cnt_j,
                                    gc_u32 *__restrict__ hit, gc_i64 words,
                                    gc_u8 *__restrict__ place,
                                    gc_i64 *__restrict__ overflow,
                                    gc_i64 npair,
                                    const gc_i64 *__restrict__ np_dev,
                                    gc_i64 pitch, double list2,
                                    float lo, float hi, float hibox,
                                    const struct gcn_cbox *__restrict__ cbox,
                                    unsigned long long *__restrict__ queue,
                                    gc_i32 *__restrict__ big)
{
    __shared__ struct gcn_sel_stage s_stage[GCN_BLOCK / 32];
    struct gcn_sel_stage &sg = s_stage[threadIdx.x >> 5];
    const int lane = threadIdx.x & 31;
    if (np_dev) npair = min(npair, *np_dev);
    for (;;) {
        unsigned long long b0 = 0;
        if (lane == 0) b0 = atomicAdd(queue, (unsigned long long)GCN_SEL_GRAB);
        b0 = __shfl_sync(0xffffffffu, b0, 0);
        if ((gc_i64)b0 >= npair) break;
        const gc_i64 pe = min((gc_i64)b0 + GCN_SEL_GRAB, npair);
        for (gc_i64 p = (gc_i64)b0; p < pe; ++p) {
            const struct gcn_pair pr = pair[p];
            const struct gcn_cell ci = cell[pr.ci];
            const struct gcn_cell cj = cell[pr.cj];
            const int ni = ci.atom_count, nj = cj.atom_count;
            int n[2] = { 0, 0 };
            if (pr.self_pair) {
                n[0] = ni; n[1] = ni;
            } else if ((gc_i64)ni > 32 * words || (gc_i64)nj > 32 * words) {
                if (lane == 0) atomicAdd((unsigned long long *)overflow, 1ull);
            } else if (ni > GCN_SEL_STAGE || nj > GCN_SEL_STAGE) {
                if (lane == 0) big[atomicAdd(queue + 1, 1ull)] = (gc_i32)p;
                continue;
            } else {
                gcn_sel_pair_warp(p, ci, cj, pr.move[0], pr.move[1], pr.move[2],
                                  coord, av, resident_slot, cbox[pr.ci],
                                  cbox[pr.cj], hit, words, place, pitch, list2,
                                  lo, hi, hibox, sg, n);
            }
            if (lane == 0) {
                if (n[0] == 0 || n[1] == 0) { n[0] = 0; n[1] = 0; }
                cnt_i[p] = n[0];
                cnt_j[p] = n[1];
            }
        }
    }
}

/* The pairs with a cell over GCN_SEL_STAGE atoms (big[0 .. queue[1])), a
 * block each, each side by gcn_sel_side; queue[2] is the block queue
 * (zeroed by the caller).  Their runs keep run order. */
__global__ void gcn_kern_sel_count_big(const gc_f64 *__restrict__ coord,
                                       const struct gcn_cell *__restrict__ cell,
                                       const gc_i32 *__restrict__ resident_slot,
                                       const struct gcn_pair *__restrict__ pair,
                                       gc_i32 *__restrict__ cnt_i,
                                       gc_i32 *__restrict__ cnt_j,
                                       gc_u32 *__restrict__ hit, gc_i64 words,
                                       gc_i64 pitch, double list2,
                                       float lo, float hi, double cmax,
                                       unsigned long long *__restrict__ queue,
                                       const gc_i32 *__restrict__ big)
{
    __shared__ gc_i64 s_next;
    __shared__ int s_n[2];
    __shared__ float s_f[3 * GCN_SEL_CHUNK];
    __shared__ int s_slot[GCN_SEL_CHUNK];
    const int lane = threadIdx.x & 31;
    const gc_i64 nbig = (gc_i64)queue[1];
    for (;;) {
        if (threadIdx.x == 0) {
            s_next = (gc_i64)atomicAdd(queue + 2, 1ull);
            s_n[0] = 0; s_n[1] = 0;
        }
        __syncthreads();
        const gc_i64 q = s_next;
        if (q >= nbig) break;
        const gc_i64 p = big[q];
        const struct gcn_cell ci = cell[pair[p].ci];
        const struct gcn_cell cj = cell[pair[p].cj];
        const double mx = pair[p].move[0], my = pair[p].move[1],
                     mz = pair[p].move[2];
        int n = gcn_sel_side(coord, resident_slot, ci, cj, mx, my, mz, pitch,
                             list2, lo, hi, cmax, hit + (2 * p) * words,
                             s_f, s_f + GCN_SEL_CHUNK, s_f + 2 * GCN_SEL_CHUNK,
                             s_slot);
        if (lane == 0 && n) atomicAdd(&s_n[0], n);
        n = gcn_sel_side(coord, resident_slot, cj, ci, -mx, -my, -mz, pitch,
                         list2, lo, hi, cmax, hit + (2 * p + 1) * words,
                         s_f, s_f + GCN_SEL_CHUNK, s_f + 2 * GCN_SEL_CHUNK,
                         s_slot);
        if (lane == 0 && n) atomicAdd(&s_n[1], n);
        __syncthreads();
        if (threadIdx.x == 0) {
            int n0 = s_n[0], n1 = s_n[1];
            if (n0 == 0 || n1 == 0) { n0 = 0; n1 = 0; }
            cnt_i[p] = n0;
            cnt_j[p] = n1;
        }
        __syncthreads();
    }
}

/* An atom's position in its pair's run on one side, from the selection's
 * bits: the place sel_count2 computed for a staged pair, else the number of
 * selected atoms before it; -1 when the selection left it out. */
__device__ __forceinline__ int gcn_run_pos(const gc_u32 *__restrict__ hit,
                                           gc_i64 words,
                                           const gc_u8 *__restrict__ place,
                                           int staged, gc_i64 p, int side,
                                           int a)
{
    const gc_u32 *row = hit + (2 * p + side) * words;
    const int w = a >> 5;
    const gc_u32 m = row[w];
    if (!((m >> (a & 31)) & 1u)) return -1;
    if (staged) return place[(2 * p + side) * 32 * words + a];
    int rank = __popc(m & ((1u << (a & 31)) - 1u));
    for (int k = 0; k < w; ++k) rank += __popc(row[k]);
    return rank;
}

/* The real-space walk's pair order: descending tile load
 * ceil(ix_n/8)*ceil(iy_n/4) (capped at 255), ties by pair index, empty pairs
 * last -- the order of GENESIS's univ_ij_sort_list (sp_pairlist_gpu.fpp),
 * so the four warps of a block carry similar work. */
__global__ void gcn_kern_pair_load(const gc_i32 *__restrict__ cnt_i,
                                   const gc_i32 *__restrict__ cnt_j,
                                   gc_i64 n, gc_u8 *__restrict__ key,
                                   gc_i32 *__restrict__ idx)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const int t = ((cnt_i[i] + 7) / 8) * ((cnt_j[i] + 3) / 4);
        key[i] = (gc_u8)(255 - min(t, 255));
        idx[i] = (gc_i32)i;
    }
}

/* Fill the runs: a pair sel_count2 staged takes the places it computed; a
 * self pair, or one with a cell over GCN_SEL_STAGE atoms, keeps run order. A
 * warp takes 32 pairs at a time, a lane per atom. */
__global__ void gcn_kern_sel_fill(const struct gcn_cell *__restrict__ cell,
                                  const gc_i32 *__restrict__ resident_slot,
                                  struct gcn_pair *__restrict__ pair,
                                  const gc_i64 *__restrict__ off_i,
                                  const gc_i64 *__restrict__ off_j,
                                  const gc_i32 *__restrict__ cnt_i,
                                  const gc_i32 *__restrict__ cnt_j,
                                  const gc_u32 *__restrict__ hit,
                                  gc_i64 words,
                                  const gc_u8 *__restrict__ place,
                                  gc_i32 *__restrict__ sel,
                                  gc_i64 npair, gc_i64 jbase)
{
    const int lane = threadIdx.x & 31;
    const gc_i64 nw = ((gc_i64)gridDim.x * blockDim.x) >> 5;
    for (gc_i64 base = (((gc_i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 32;
         base < npair; base += 32 * nw) {
        const gc_i64 q = base + lane;
        int n0 = 0, self = 0, ib = 0, ni = 0, jb = 0, nj = 0;
        gc_i64 bi = 0, bj = 0;
        if (q < npair) {
            n0 = cnt_i[q];
            if (n0 == 0) {
                pair[q].ix_n = 0; pair[q].iy_n = 0;
            } else {
                const struct gcn_cell ci = cell[pair[q].ci];
                const struct gcn_cell cj = cell[pair[q].cj];
                self = pair[q].self_pair;
                ib = ci.atom_begin; ni = ci.atom_count;
                jb = cj.atom_begin; nj = cj.atom_count;
                bi = off_i[q]; bj = jbase + off_j[q];
                pair[q].ix_off = (gc_i32)bi;
                pair[q].iy_off = (gc_i32)bj;
                pair[q].ix_n   = n0;
                pair[q].iy_n   = cnt_j[q];
            }
        }
        for (gc_u32 live = __ballot_sync(0xffffffffu, n0 != 0); live;
             live &= live - 1) {
            const int k = __ffs(live) - 1;
            const gc_i64 p = base + k;
            const int ps = __shfl_sync(0xffffffffu, self, k);
            const int pib = __shfl_sync(0xffffffffu, ib, k);
            const int pni = __shfl_sync(0xffffffffu, ni, k);
            const int pjb = __shfl_sync(0xffffffffu, jb, k);
            const int pnj = __shfl_sync(0xffffffffu, nj, k);
            const gc_i64 pbi = __shfl_sync(0xffffffffu, bi, k);
            const gc_i64 pbj = __shfl_sync(0xffffffffu, bj, k);
            if (ps) {
                for (int a = lane; a < pni; a += 32) {
                    sel[pbi + a] = resident_slot[pib + a];
                    sel[pbj + a] = resident_slot[pib + a];
                }
                continue;
            }
            const int staged = pni <= GCN_SEL_STAGE && pnj <= GCN_SEL_STAGE;
            if (staged && words <= 16) {
                /* only the words sel_count2 wrote: ceil(n/32) per side */
                const int hs = lane >= (int)words;
                const int hk = lane - hs * (int)words;
                const gc_u32 hw = (lane < 2 * words && 32 * hk < (hs ? pnj : pni))
                                      ? hit[2 * p * words + lane] : 0u;
                const gc_u8 *pl = place + 2 * p * 32 * words;
#pragma unroll 4
                for (int t0 = 0; t0 < pni + pnj; t0 += 32) {
                    const int t = t0 + lane;
                    const int side = t < pni ? 0 : 1;
                    const int a = side ? t - pni : t;
                    const gc_u32 w = __shfl_sync(0xffffffffu, hw,
                                                 (side * (int)words + (a >> 5)) & 31);
                    if (t < pni + pnj && ((w >> (a & 31)) & 1u)) {
                        const int rank = pl[side * 32 * words + a];
                        sel[(side ? pbj : pbi) + rank] =
                            resident_slot[(side ? pjb : pib) + a];
                    }
                }
                continue;
            }
            for (int t = lane; t < pni + pnj; t += 32) {
                const int side = t < pni ? 0 : 1;
                const int a = side ? t - pni : t;
                const int rank = gcn_run_pos(hit, words, place, staged, p, side, a);
                if (rank < 0) continue;
                sel[(side ? pbj : pbi) + rank] = resident_slot[(side ? pjb : pib) + a];
            }
        }
    }
}

/* The FP32 prefilter's decision bounds for the selection test.  With
 * u = 2^-24, every coordinate the FP32 path sees is bounded by cmax = 16 box
 * edges + 64 A; atoms beyond it take the FP64 test with a NaN copy.  Each FP32
 * copy is off by at most u*cmax, the difference vector by
 * e(D) <= sqrt(3) * (2u*cmax + u*(D + 2u*cmax)) at distance D, and the
 * squared sum carries a further relative error below 4u.  Doubling every
 * error term:
 *
 *   d2f < lo = (1-8u) (R - 2 e(R))^2             proves D < R,
 *   d2f > hi = (1+8u) (R (1+2 sqrt3 u) + 2 e0)^2  proves D > R, e0 = e(0).
 *
 * lo is rounded down and hi up when narrowed to float. */
void gcn_sel_bounds(double list2, const double *box,
                           float *lo_out, float *hi_out, double *cmax_out)
{
    const double u = std::ldexp(1.0, -24);
    double bmax = box[0];
    if (box[1] > bmax) bmax = box[1];
    if (box[2] > bmax) bmax = box[2];
    const double cmax = 4.0 * bmax + 64.0;
    *cmax_out = cmax;
    const double R = std::sqrt(list2);
    const double s3 = std::sqrt(3.0);
    /* 6u cmax, not 2u cmax: a moved view is (float)coord + (float)move,
     * within 3u cmax of the FP64 value, and a partner view within u cmax / 2. */
    const double eR = s3 * (6.0 * u * cmax + u * (R + 6.0 * u * cmax));
    const double e0 = s3 * (6.0 * u * cmax + u * (6.0 * u * cmax));
    double lo = (R > 2.0 * eR) ? (1.0 - 8.0 * u) * (R - 2.0 * eR) * (R - 2.0 * eR)
                               : 0.0;
    double hi = (1.0 + 8.0 * u) * (R * (1.0 + 2.0 * s3 * u) + 2.0 * e0)
                                * (R * (1.0 + 2.0 * s3 * u) + 2.0 * e0);
    float lf = (float)lo;
    if ((double)lf > lo) lf = std::nextafter(lf, 0.0f);
    float hf = (float)hi;
    if ((double)hf < hi) hf = std::nextafter(hf, std::numeric_limits<float>::infinity());
    *lo_out = lf;
    *hi_out = hf;
}

__device__ __forceinline__ int gcn_mask_runs_ok(gc_i32 ni, gc_i32 nj,
                                                gc_i32 ai, gc_i32 aj)
{
    return ni >= 0 && nj >= 0 && ni <= ai && nj <= aj;
}

/* What the mask kernel does with a term whose pair entry is p:
 *
 *   CLEAR  the pair carries a mask: clear the term's bit;
 *   SKIP   valid run shape with one run empty: the real-space kernel walks no
 *          atom pair, so there is no bit and nothing is counted twice;
 *   FAIL   no entry, an unwalkable run shape, or atoms on both sides with no
 *          mask (mask_off -1 leaves every interaction active: a double count).
 */
enum {
    GCN_MASK_DO_CLEAR = 0,
    GCN_MASK_DO_SKIP  = 1,
    GCN_MASK_DO_FAIL  = 2
};

__device__ __forceinline__ int
gcn_mask_pair_verdict(const struct gcn_pair *pair, const struct gcn_cell *cell,
                      gc_i32 p)
{
    if (p < 0) return GCN_MASK_DO_FAIL;
    const gc_i32 ni = pair[p].ix_n, nj = pair[p].iy_n;
    if (!gcn_mask_runs_ok(ni, nj, cell[pair[p].ci].atom_count,
                          cell[pair[p].cj].atom_count))
        return GCN_MASK_DO_FAIL;
    if (ni == 0 || nj == 0) return GCN_MASK_DO_SKIP;
    if (pair[p].mask_off < 0) return GCN_MASK_DO_FAIL;
    return GCN_MASK_DO_CLEAR;
}

/* Mask sizing and initialisation. A mask exists only for a pair whose cells
 * are within one cell of each other, the only place an excluded or 1-4 pair
 * can live; the scatter refuses a term that lands elsewhere. The bit count of
 * a masked pair is 64-bit (a 32-bit product of run lengths would wrap and the
 * scatter would run off the allocation); run lengths are also checked against
 * their cells here. */
__global__ void gcn_kern_mask_count(const struct gcn_pair *__restrict__ pair,
                                    const gc_i32 *__restrict__ cnt_i,
                                    const gc_i32 *__restrict__ cnt_j,
                                    const struct gcn_cell *__restrict__ cell,
                                    gc_i64 *__restrict__ bits,
                                    gc_i64 npair,
                                    const gc_i64 *__restrict__ np_dev,
                                    struct gcn_stencil st,
                                    gc_i64 *__restrict__ rep)
{
    if (np_dev) npair = min(npair, *np_dev);
    for (gc_i64 p = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; p < npair;
         p += (gc_i64)gridDim.x * blockDim.x) {
        int near = 1;
        int d;
        d = cell[pair[p].ci].gx - cell[pair[p].cj].gx;
        if (d >  st.ncel[0]/2) d -= st.ncel[0];
        if (d < -st.ncel[0]/2) d += st.ncel[0];
        if (d < -1 || d > 1) near = 0;
        d = cell[pair[p].ci].gy - cell[pair[p].cj].gy;
        if (d >  st.ncel[1]/2) d -= st.ncel[1];
        if (d < -st.ncel[1]/2) d += st.ncel[1];
        if (d < -1 || d > 1) near = 0;
        d = cell[pair[p].ci].gz - cell[pair[p].cj].gz;
        if (d >  st.ncel[2]/2) d -= st.ncel[2];
        if (d < -st.ncel[2]/2) d += st.ncel[2];
        if (d < -1 || d > 1) near = 0;
        /* the run lengths sel_fill publishes (ix_n = cnt_i, iy_n = cnt_j,
           both zero when either is), read before the fill */
        const gc_i32 ni = cnt_i[p];
        const gc_i32 nj = cnt_j[p];
        if (!gcn_mask_runs_ok(ni, nj, cell[pair[p].ci].atom_count,
                              cell[pair[p].cj].atom_count)) {
            if (atomicAdd((unsigned long long *)&rep[0], 1ull) == 0ull) {
                rep[1] = p; rep[2] = ni; rep[3] = nj;
            }
            bits[p] = 0;
            continue;
        }
        bits[p] = (near && ni > 0 && nj > 0) ? (gc_i64)ni * (gc_i64)nj : 0;
    }
}

/* Every bit of a masked pair starts set, except that a self pair keeps only
 * the upper triangle. The pairs' bit ranges are the exclusive scan of their
 * bit counts, so they tile [0, total): gcn_kern_mask_fill sets every bit
 * below total (and clears the words above), and gcn_kern_mask_set writes each
 * pair's offset and clears the self pairs' lower triangles. */
__global__ void gcn_kern_mask_fill(gc_u64 *__restrict__ m, gc_i64 n,
                                   gc_i64 total_bits)
{
    const gc_i64 full = total_bits >> 6;
    const int tail = (int)(total_bits & 63);
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x)
        m[i] = i < full ? ~0ull
             : (i == full && tail) ? (1ull << tail) - 1ull : 0ull;
}

__device__ __forceinline__ void gcn_mask_self(gc_i64 off, gc_i64 ni, gc_i64 nj,
                                              const gc_i32 *__restrict__ ix,
                                              const gc_i32 *__restrict__ iy,
                                              gc_u64 *__restrict__ mask,
                                              int tid, int nth)
{
    const gc_i64 end = off + ni * nj;
    const gc_i64 w0 = off >> 6, w1 = (end - 1) >> 6;
    for (gc_i64 w = w0 + tid; w <= w1; w += nth) {
        const gc_i64 lo = (w << 6) > off ? (w << 6) : off;
        const gc_i64 hi = ((w + 1) << 6) < end ? ((w + 1) << 6) : end;
        gc_u64 m = 0ull;
        gc_i64 k = lo - off;
        gc_i64 a = k / nj, b = k - a * nj;
        for (gc_i64 bit = lo; bit < hi; ++bit) {
            if (ix[a] < iy[b]) m |= 1ull << (bit & 63);
            if (++b == nj) { b = 0; ++a; }
        }
        if (lo == (w << 6) && hi == ((w + 1) << 6)) {
            mask[w] = m;
        } else {
            const int n = (int)(hi - lo), b0 = (int)(lo & 63);
            const gc_u64 range = (n == 64) ? ~0ull : (((1ull << n) - 1ull) << b0);
            atomicAnd((unsigned long long *)&mask[w],
                      (unsigned long long)(m | ~range));
        }
    }
}

/* Every pair's mask offset (-1 for a pair with no mask), refusing once, with
 * the numbers that disagree, a range past the mask or a self pair's runs past
 * the selection. A warp takes 32 pairs at a time. */
__global__ void gcn_kern_mask_set(struct gcn_pair *__restrict__ pair,
                                  const gc_i64 *__restrict__ bits,
                                  const gc_i64 *__restrict__ off_m,
                                  const gc_i32 *__restrict__ sel,
                                  gc_u64 *__restrict__ mask, gc_i64 npair,
                                  gc_i64 mask_words, gc_i64 sel_words,
                                  gc_i64 *__restrict__ rep)
{
    const int lane = threadIdx.x & 31;
    const gc_i64 nw = ((gc_i64)gridDim.x * blockDim.x) >> 5;
    for (gc_i64 base = (((gc_i64)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 32;
         base < npair; base += 32 * nw) {
        const gc_i64 q = base + lane;
        gc_i64 off = -1, ni = 0, nj = 0, ixo = 0, iyo = 0;
        int self = 0;
        if (q < npair) {
            off = bits[q] == 0 ? -1 : off_m[q];
            pair[q].mask_off = off;
            if (off >= 0) {
                ni = pair[q].ix_n; nj = pair[q].iy_n;
                self = pair[q].self_pair;
                ixo = pair[q].ix_off; iyo = pair[q].iy_off;
                const gc_i64 last = off + ni * nj - 1;
                int bad = 0;
                if (ni > 0 && nj > 0 && (last >> 6) >= mask_words) bad = 1;
                else if (self && (ixo + ni > sel_words || iyo + nj > sel_words ||
                                  ixo < 0 || iyo < 0)) bad = 2;
                if (bad) {
                    if (atomicAdd((unsigned long long *)&rep[4], 1ull) == 0ull) {
                        rep[5] = q;
                        rep[6] = bad == 1 ? ni * nj : -(ni * nj);
                        rep[7] = bad == 1 ? off : ixo;
                    }
                    self = 0;
                }
            }
        }
        for (gc_u32 live = __ballot_sync(0xffffffffu, self && ni > 0 && nj > 0);
             live; live &= live - 1) {
            const int k = __ffs(live) - 1;
            gcn_mask_self(__shfl_sync(0xffffffffu, off, k),
                          __shfl_sync(0xffffffffu, ni, k),
                          __shfl_sync(0xffffffffu, nj, k),
                          sel + __shfl_sync(0xffffffffu, ixo, k),
                          sel + __shfl_sync(0xffffffffu, iyo, k),
                          mask, lane, 32);
        }
    }
}

/* The pair half of the list statistics across the whole grid: report[8]
 * the masked pairs, [9] and [10] the longest selected runs (the caller
 * zeroes the report; the cell half is the cell scan's). */
__global__ void gcn_kern_list_stats2(const gc_i64 *__restrict__ bits,
                                     const gc_i32 *__restrict__ cnt_i,
                                     const gc_i32 *__restrict__ cnt_j,
                                     gc_i64 npair,
                                     const gc_i64 *__restrict__ np_dev,
                                     gc_i64 *__restrict__ out)
{
    if (np_dev) npair = min(npair, *np_dev);
    gc_i64 n = 0; gc_i32 mi = 0, mj = 0;
    for (gc_i64 p = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; p < npair;
         p += (gc_i64)gridDim.x * blockDim.x) {
        if (bits[p] > 0) ++n;
        mi = max(mi, cnt_i[p]); mj = max(mj, cnt_j[p]);
    }
    for (int o = 16; o > 0; o >>= 1) {
        n += __shfl_xor_sync(0xffffffffu, n, o);
        mi = max(mi, __shfl_xor_sync(0xffffffffu, mi, o));
        mj = max(mj, __shfl_xor_sync(0xffffffffu, mj, o));
    }
    if ((threadIdx.x & 31) == 0) {
        if (n)  atomicAdd((unsigned long long *)&out[8], (unsigned long long)n);
        if (mi) atomicMax((unsigned long long *)&out[9], (unsigned long long)mi);
        if (mj) atomicMax((unsigned long long *)&out[10], (unsigned long long)mj);
    }
}

/* Execution term records */

/* The image code used for pair terms without stock per-edge codes (bonded
 * terms use their imported codes). A ghost holds its owner's raw coordinates,
 * so the code is taken between the wrapped global cells, not the extended
 * ones. */
__device__ __forceinline__ int gcn_pbc_code(const struct gcn_cell *cell,
                                            const gc_i32 *cell_of,
                                            int a, int b, const int *ncel)
{
    int k[3], w;
    const struct gcn_cell ca = cell[cell_of[a]], cb = cell[cell_of[b]];
    const int ga[3] = { ca.gx, ca.gy, ca.gz };
    const int gb[3] = { cb.gx, cb.gy, cb.gz };
    for (int d = 0; d < 3; ++d) {
        int dd = gcn_wrap_axis(ga[d], ncel[d], &w)
               - gcn_wrap_axis(gb[d], ncel[d], &w);
        k[d] = 0;
        if (dd >  ncel[d] / 2) k[d] = -1;
        if (dd < -ncel[d] / 2) k[d] = +1;
    }
    return (k[2] + 1) * 9 + (k[1] + 1) * 3 + (k[0] + 1);
}

__device__ __forceinline__ int gcn_cell_cheb(const struct gcn_cell *cell,
                                             const gc_i32 *cell_of, int a, int b)
{
    const struct gcn_cell ca = cell[cell_of[a]], cb = cell[cell_of[b]];
    return max(max(abs(ca.gx - cb.gx), abs(ca.gy - cb.gy)), abs(ca.gz - cb.gz));
}

__global__ void gcn_kern_exec_compare(const struct gcn_term_exec *__restrict__ x,
                                      const struct gcn_term_exec *__restrict__ y,
                                      gc_i64 n, gc_i64 *__restrict__ bad)
{
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (gc_i64)gridDim.x * blockDim.x) {
        const int *a = (const int *)&x[t], *b = (const int *)&y[t];
        int diff = 0;
        for (int w = 0; w < (int)(sizeof(struct gcn_term_exec) / 4); ++w)
            diff |= a[w] != b[w];
        if (diff) atomicAdd((unsigned long long *)bad, 1ull);
    }
}

__global__ void gcn_kern_invert_perm(const gc_i32 *__restrict__ perm,
                                     gc_i64 n, gc_i32 *__restrict__ inv,
                                     gc_i64 *__restrict__ bad)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i32 o = perm[i];
        if (o < 0 || o >= n) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
        inv[o] = (gc_i32)i;
    }
}

/* For every old slot, its new slot and the new slot's cell coordinates
 * packed 10 bits each (the carry requires fewer than 1024 cells a side):
 * one 8-byte read per endpoint serves the record and its codes. */
__global__ void gcn_kern_carry_map(const gc_i32 *__restrict__ inv, gc_i64 n,
                                   const gc_i32 *__restrict__ cell_of,
                                   const struct gcn_cell *__restrict__ cell,
                                   int2 *__restrict__ pk,
                                   gc_i64 *__restrict__ bad)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i32 ns = inv[i];
        const struct gcn_cell c = cell[cell_of[ns]];
        if ((unsigned)c.gx > 1023u || (unsigned)c.gy > 1023u ||
            (unsigned)c.gz > 1023u)
            atomicAdd((unsigned long long *)bad, 1ull);
        pk[i] = make_int2(ns, c.gx | (c.gy << 10) | (c.gz << 20));
    }
}

__device__ __forceinline__ int gcn_pk_code(int ga, int gb, const int *ncel)
{
    int k[3], w;
    for (int d = 0; d < 3; ++d) {
        const int dd = gcn_wrap_axis((ga >> (10 * d)) & 1023, ncel[d], &w)
                     - gcn_wrap_axis((gb >> (10 * d)) & 1023, ncel[d], &w);
        k[d] = 0;
        if (dd >  ncel[d] / 2) k[d] = -1;
        if (dd < -ncel[d] / 2) k[d] = +1;
    }
    return (k[2] + 1) * 9 + (k[1] + 1) * 3 + (k[0] + 1);
}

/* Resolve one kind's canonical records into execution records.  A missing
 * endpoint is an error, never a dropped term: `bad` is raised and the
 * rebuild refuses. */
__global__ void gcn_kern_term_exec(const gc_gid *__restrict__ endpoint,
                                   const gc_image *__restrict__ endpoint_image,
                                   const gc_i32 *__restrict__ param,
                                   const gc_i32 *__restrict__ source_pbc,
                                   gc_i64 count, int arity,
                                   const gc_gid *__restrict__ mkey,
                                   const gc_image *__restrict__ mikey,
                                   const gc_i32 *__restrict__ mval,
                                   gc_i64 mcap,
                                   const struct gcn_cell *__restrict__ cell,
                                   const gc_i32 *__restrict__ cell_of,
                                   struct gcn_term_exec *__restrict__ out,
                                   gc_i64 *__restrict__ bad,
                                   int nx, int ny, int nz, int kind,
                                   int ncodes)
{
    int ncel[3] = { nx, ny, nz };
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; t < count;
         t += (gc_i64)gridDim.x * blockDim.x) {

        int a[GC_MAX_TERM_ARITY];
        int ok = 1, ref = -1;
        for (int e = 0; e < arity; ++e) {
            a[e] = gcn_imap_find(mkey, mikey, mval, mcap,
                                  endpoint[t * arity + e],
                                  endpoint_image[t * arity + e]);
            if (a[e] >= 0 && ref < 0) ref = e;
        }
        /* After a migration an endpoint's recorded copy may be gone or the
         * far one; every copy holds the owner's raw coordinates, so take the
         * copy nearest the term's first resolved endpoint. */
        for (int e = 0; e < arity; ++e) {
            if (e == ref) continue;
            if (a[e] >= 0 && gcn_cell_cheb(cell, cell_of, a[e], a[ref]) <= 2)
                continue;
            int best = -1, bd = 1 << 30;
            for (int q = 0; q < 27; ++q) {
                const gc_image im =
                    ((gc_image)(gc_i16)(q % 3 - 1) & 0xffffLL) |
                    (((gc_image)(gc_i16)((q / 3) % 3 - 1) & 0xffffLL) << 16) |
                    (((gc_image)(gc_i16)(q / 9 - 1) & 0xffffLL) << 32);
                const int s = gcn_imap_find(mkey, mikey, mval, mcap,
                                            endpoint[t * arity + e], im);
                if (s < 0) continue;
                const int dd = ref < 0 ? 0 : gcn_cell_cheb(cell, cell_of, s, a[ref]);
                if (dd < bd) { bd = dd; best = s; }
            }
            a[e] = best;
            if (best < 0) ok = 0;
            else if (ref < 0) ref = e;
        }
        if (!ok) { atomicAdd((unsigned long long *)bad, 1ull); continue; }

        for (int e = 0; e < arity; ++e) out[t].a[e] = a[e];
        for (int e = arity; e < GC_MAX_TERM_ARITY; ++e) out[t].a[e] = a[0];
        for (int e = 0; e < 6; ++e) out[t].pbc[e] = 13;   /* identity */
        if (source_pbc) {
            for (int e = 0; e < ncodes; ++e)
                out[t].pbc[e] = source_pbc[t * ncodes + e];
        }
        out[t].param = param ? param[t] : 0;
        out[t].pad0  = 0;

        if (source_pbc) continue;
        switch (kind) {
          case GC_TERM_BOND:
          case GC_TERM_NB14:
          case GC_TERM_EXCL:
            out[t].pbc[0] = gcn_pbc_code(cell, cell_of, a[0], a[1], ncel);
            break;
          case GC_TERM_ANGLE:
            /* d12, d32 and, for Urey-Bradley, d13 */
            out[t].pbc[0] = gcn_pbc_code(cell, cell_of, a[0], a[1], ncel);
            out[t].pbc[1] = gcn_pbc_code(cell, cell_of, a[2], a[1], ncel);
            out[t].pbc[2] = gcn_pbc_code(cell, cell_of, a[0], a[2], ncel);
            break;
          case GC_TERM_DIHEDRAL:
          case GC_TERM_IMPROPER:
            /* dij = 1-2, djk = 2-3, dlk = 4-3 */
            out[t].pbc[0] = gcn_pbc_code(cell, cell_of, a[0], a[1], ncel);
            out[t].pbc[1] = gcn_pbc_code(cell, cell_of, a[1], a[2], ncel);
            out[t].pbc[2] = gcn_pbc_code(cell, cell_of, a[3], a[2], ncel);
            break;
          case GC_TERM_CMAP:
            out[t].pbc[0] = gcn_pbc_code(cell, cell_of, a[0], a[1], ncel);
            out[t].pbc[1] = gcn_pbc_code(cell, cell_of, a[1], a[2], ncel);
            out[t].pbc[2] = gcn_pbc_code(cell, cell_of, a[3], a[2], ncel);
            out[t].pbc[3] = gcn_pbc_code(cell, cell_of, a[4], a[5], ncel);
            out[t].pbc[4] = gcn_pbc_code(cell, cell_of, a[5], a[6], ncel);
            out[t].pbc[5] = gcn_pbc_code(cell, cell_of, a[7], a[6], ncel);
            break;
          default: break;
        }
    }
}

enum {
    GCN_MASK_POOL_REAL = 0,   /* the imported real-mask descriptor pairs */
    GCN_MASK_POOL_NB14 = 1,   /* the 1-4 term pool                       */
    GCN_MASK_POOL_EXCL = 2,   /* the exclusion term pool                 */
    GCN_MASK_POOL_N    = 3
};
enum {
    GCN_MASK_COND_CELL    = 1,  /* cell_of outside the box             */
    GCN_MASK_COND_STENCIL = 2,  /* endpoint cells beyond the stencil   */
    GCN_MASK_COND_PAIR    = 3,  /* no list entry, or no mask for it    */
    GCN_MASK_COND_BIT     = 4,  /* computed bit outside the mask words */
    GCN_MASK_COND_N       = 5
};

/* Clear the mask bit of every excluded and 1-4 pair, so the real-space kernel
 * skips what the correction and 1-4 kernels own. A pair whose cells are not
 * adjacent has no mask: an error, since the interaction would be counted
 * twice. */

__device__ __forceinline__ int gcn_cell_index(const gc_i32 *__restrict__ resident_slot,
                                              const struct gcn_cell &c, gc_i32 s)
{
    int lo = 0, hi = c.atom_count;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (resident_slot[c.atom_begin + mid] < s) lo = mid + 1;
        else hi = mid;
    }
    return lo < c.atom_count && resident_slot[c.atom_begin + lo] == s ? lo : -1;
}

struct gcn_mask_args {
    const gc_i32 *cell_of;
    const struct gcn_cell *cell;
    const struct gcn_pair *pair;
    const gc_i32 *pair_at;
    const gc_i32 *sel;
    const gc_i32 *resident_slot;
    const gc_u32 *hit;
    gc_i64 words;
    const gc_u8 *place;
    gc_u64 *mask;
    gc_i64 *bad;
    gc_i64 mask_words;
    struct gcn_stencil st;
    int4 *xl;
    int *xn;
    gc_i64 xcap;
};

__device__ __forceinline__ void gcn_mask_clear(const struct gcn_mask_args &m,
                                               const struct gcn_term_exec *t,
                                               gc_i64 count, gc_i64 i,
                                               int sa, int sb, gc_i32 pool)
{
    gc_i32 ca = m.cell_of[sa], cb = m.cell_of[sb];
    if (ca < 0 || cb < 0 || ca >= m.st.layout.ncell_box ||
        cb >= m.st.layout.ncell_box) {
        atomicAdd((unsigned long long *)m.bad, 1ull);
        return;
    }

    /* The pair's home cell and the enumerator's displacement (one rule), so
     * the lookup lands on the pair the real-space pass walks. */
    gc_i32 home = -1, other = -1;
    gc_i32 d[3];
    home = gcn_mask_home_other(m.st, m.cell, ca, cb, &other, d);
    if (home < 0) {
        return;
    }
    if (home != ca) { const int w = sa; sa = sb; sb = w; }
    int ok = 1;
    for (int k = 0; k < 3; ++k) {
        if (d[k] < -m.st.nh[k] || d[k] > m.st.nh[k]) {
            atomicAdd((unsigned long long *)m.bad, 1ull);
            ok = 0; break;
        }
    }
    if (!ok) return;

    gc_i32 p = m.pair_at[(gc_i64)home * m.st.size
                         + gcn_ordinal_of(m.st, d[0], d[1], d[2])];
    /* The cluster-only list (no runs, no mask; k_nbc_flags): the pair's
     * cluster word bit is cleared, the i side's slot first. A missing pair is
     * the refusal below. */
    if (!m.sel && p >= 0) {
        if (m.pair[p].ix_n == 0 || !m.xl) return;
        if (gcn_cell_index(m.resident_slot, m.cell[m.pair[p].ci], sa) < 0 ||
            gcn_cell_index(m.resident_slot, m.cell[m.pair[p].cj], sb) < 0)
            return;
        if (m.pair[p].self_pair && sa > sb) { const int w = sa; sa = sb; sb = w; }
        const int k = atomicAdd(m.xn, 1);
        if (k < m.xcap) m.xl[k] = make_int4(sa, sb, (int)p, -1);
        return;
    }
    const int verdict = gcn_mask_pair_verdict(m.pair, m.cell, p);
    if (verdict == GCN_MASK_DO_FAIL) {
        atomicAdd((unsigned long long *)m.bad, 1ull);
        return;
    }
    if (verdict == GCN_MASK_DO_SKIP) {
        return;
    }

    const gc_i32 *ix = m.sel + m.pair[p].ix_off;
    const gc_i32 *iy = m.sel + m.pair[p].iy_off;
    const int nj = m.pair[p].iy_n;

    const struct gcn_cell hc = m.cell[m.pair[p].ci], oc = m.cell[m.pair[p].cj];
    const int a = gcn_cell_index(m.resident_slot, hc, sa);
    const int b = gcn_cell_index(m.resident_slot, oc, sb);
    int ia = -1, jb = -1;
    if (a >= 0 && b >= 0) {
        if (m.pair[p].self_pair) {
            ia = a;
            jb = b;
        } else {
            const int staged = hc.atom_count <= GCN_SEL_STAGE &&
                               oc.atom_count <= GCN_SEL_STAGE;
            ia = gcn_run_pos(m.hit, m.words, m.place, staged, p, 0, a);
            jb = gcn_run_pos(m.hit, m.words, m.place, staged, p, 1, b);
        }
    }

    /* A self pair keeps only the upper triangle, so the bit to clear
       is the one that was set: (min, max) in slot order. */
    if (m.pair[p].self_pair && ia >= 0 && jb >= 0 && ix[ia] > iy[jb]) {
        int s = ia; ia = jb; jb = s;
    }

    if (ia >= 0 && jb >= 0) {
        gc_i64 bit = m.pair[p].mask_off + (gc_i64)ia * nj + jb;
        if ((bit >> 6) >= m.mask_words || bit < 0) {
            atomicAdd((unsigned long long *)m.bad, 1ull);
            return;
        }
        atomicAnd((unsigned long long *)&m.mask[bit >> 6],
                  ~(1ull << (bit & 63)));
        if (m.xl) {
            const int k = atomicAdd(m.xn, 1);
            if (k < m.xcap) m.xl[k] = make_int4(ix[ia], iy[jb], (int)p, -1);
        }
    }
    /* Not being in either run means the two atoms are farther apart
       than the list radius, so the real-space kernel never visits the
       pair and there is nothing to mask. */
}

/* Record arrays handled by one launch: segment k is rec[k][0 .. start[k+1]
 * - start[k]), of term kind kind[k] with arity[k] endpoints, cell codes
 * recomputed when codes[k], and mask pool pool[k] (-1: none). */
#define GCN_TERM_SEGS 8
struct gcn_term_segs {
    struct gcn_term_exec *rec[GCN_TERM_SEGS];
    gc_i64 start[GCN_TERM_SEGS + 1];
    int kind[GCN_TERM_SEGS], arity[GCN_TERM_SEGS], codes[GCN_TERM_SEGS];
    int pool[GCN_TERM_SEGS];
    int n;
};

__device__ __forceinline__ int gcn_seg_of(const struct gcn_term_segs &g,
                                        gc_i64 x)
{
    int k = 0;
    while (k + 1 < g.n && x >= g.start[k + 1]) ++k;
    return k;
}

static void gcn_seg_add(struct gcn_term_segs &g, struct gcn_term_exec *rec,
                        gc_i64 count, int kind, int arity, int codes, int pool)
{
    if (count <= 0) return;
    if (g.n == 0) g.start[0] = 0;
    g.rec[g.n] = rec; g.kind[g.n] = kind; g.arity[g.n] = arity;
    g.codes[g.n] = codes; g.pool[g.n] = pool;
    g.start[g.n + 1] = g.start[g.n] + count;
    ++g.n;
}

__global__ void gcn_kern_mask_exclusions(struct gcn_term_segs g,
                                         struct gcn_mask_args m)
{
    const gc_i64 n = g.start[g.n];
    for (gc_i64 x = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; x < n;
         x += (gc_i64)gridDim.x * blockDim.x) {
        const int k = gcn_seg_of(g, x);
        const gc_i64 i = x - g.start[k];
        const struct gcn_term_exec *t = g.rec[k];
        gcn_mask_clear(m, t, g.start[k + 1] - g.start[k], i,
                       t[i].a[0], t[i].a[1], g.pool[k]);
    }
}

/* The last build's records carried to this build's slots, every segment in
 * one launch: endpoint slots through the map, cell-derived codes
 * (gcn_pbc_code) recomputed; the record equals gcn_kern_term_exec's for the
 * same atoms. A segment with a mask pool then clears its mask bit as
 * gcn_kern_mask_exclusions does. */
__global__ void gcn_kern_term_remap(const int2 *__restrict__ pk, gc_i64 nslot,
                                    struct gcn_term_segs g,
                                    int nx, int ny, int nz,
                                    gc_i64 *__restrict__ bad,
                                    struct gcn_mask_args m)
{
    const int ncel[3] = { nx, ny, nz };
    const gc_i64 n = g.start[g.n];
    for (gc_i64 x = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; x < n;
         x += (gc_i64)gridDim.x * blockDim.x) {
        const int sg = gcn_seg_of(g, x);
        const gc_i64 t = x - g.start[sg];
        struct gcn_term_exec *rec = g.rec[sg];
        const int arity = g.arity[sg], kind = g.kind[sg];
        int4 *r4 = (int4 *)&rec[t];
        const int4 v0 = r4[0];
        const int4 v1 = r4[1];
        const int old[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
        int a[8], gc[8];
        int ok = 1;
        for (int e = 0; e < 8; ++e) {
            if (e < arity) {
                const gc_i32 o = old[e];
                if (o < 0 || o >= nslot) { ok = 0; a[e] = 0; gc[e] = 0; }
                else { const int2 q = pk[o]; a[e] = q.x; gc[e] = q.y; }
            } else { a[e] = a[0]; gc[e] = gc[0]; }
        }
        if (!ok) { atomicAdd((unsigned long long *)bad, 1ull); continue; }
        r4[0] = make_int4(a[0], a[1], a[2], a[3]);
        r4[1] = make_int4(a[4], a[5], a[6], a[7]);
        if (g.codes[sg]) {
            int c[6] = { 13, 13, 13, 13, 13, 13 };
            switch (kind) {
              case GC_TERM_BOND:
              case GC_TERM_NB14:
              case GC_TERM_EXCL:
                c[0] = gcn_pk_code(gc[0], gc[1], ncel);
                break;
              case GC_TERM_ANGLE:
                c[0] = gcn_pk_code(gc[0], gc[1], ncel);
                c[1] = gcn_pk_code(gc[2], gc[1], ncel);
                c[2] = gcn_pk_code(gc[0], gc[2], ncel);
                break;
              case GC_TERM_DIHEDRAL:
              case GC_TERM_IMPROPER:
                c[0] = gcn_pk_code(gc[0], gc[1], ncel);
                c[1] = gcn_pk_code(gc[1], gc[2], ncel);
                c[2] = gcn_pk_code(gc[3], gc[2], ncel);
                break;
              case GC_TERM_CMAP:
                c[0] = gcn_pk_code(gc[0], gc[1], ncel);
                c[1] = gcn_pk_code(gc[1], gc[2], ncel);
                c[2] = gcn_pk_code(gc[3], gc[2], ncel);
                c[3] = gcn_pk_code(gc[4], gc[5], ncel);
                c[4] = gcn_pk_code(gc[5], gc[6], ncel);
                c[5] = gcn_pk_code(gc[7], gc[6], ncel);
                break;
              default: break;
            }
            r4[2] = make_int4(c[0], c[1], c[2], c[3]);
            if (kind == GC_TERM_CMAP) { rec[t].pbc[4] = c[4]; rec[t].pbc[5] = c[5]; }
        }
        if (g.pool[sg] >= 0)
            gcn_mask_clear(m, rec, g.start[sg + 1] - g.start[sg], t,
                           a[0], a[1], g.pool[sg]);
    }
}

__global__ void gcn_kern_term_pair(const struct gcn_term_exec *__restrict__ t,
                                   gc_i64 n, struct gcn_term_pair *__restrict__ p)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        struct gcn_term_pair q;
        q.a0 = t[i].a[0]; q.a1 = t[i].a[1]; q.pbc = t[i].pbc[0]; q.param = t[i].param;
        p[i] = q;
    }
}

/* The list-validity guard */

/* Per block, the largest squared distance of an owned atom from the
 * position the list was built from (part[b]) and from its position at the
 * start of the step (part[nblk + b]), with consistent periodic images.
 * A non-finite distance counts as +infinity: the certificate is invalid,
 * the guard rebuilds and the flush reports it, never zero motion. */
__global__ void gcn_kern_displacement(const gc_f64 *__restrict__ coord,
                                      const gc_f64 *__restrict__ ref,
                                      const gc_f64 *__restrict__ prev,
                                      gc_i64 n, gc_i64 pitch,
                                      double bx, double by, double bz,
                                      gc_f64 *__restrict__ part)
{
    __shared__ double sh[2][GCN_RED_BLOCK];
    double m = 0.0, ms = 0.0;
    double box[3] = { bx, by, bz };

    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n;
         s += (gc_i64)gridDim.x * blockDim.x) {
        double x[3], r[3], p[3], d2, s2;
        for (int c = 0; c < 3; ++c) {
            x[c] = coord[(gc_i64)c * pitch + s];
            r[c] = ref[(gc_i64)c * pitch + s];
            p[c] = prev[(gc_i64)c * pitch + s];
        }
        guard_d2(x, r, p, box, &d2, &s2);
        if (d2 > m) m = d2;
        if (s2 > ms) ms = s2;
    }
    sh[0][threadIdx.x] = m;
    sh[1][threadIdx.x] = ms;
    __syncthreads();
    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if ((int)threadIdx.x < half) {
            for (int k = 0; k < 2; ++k)
                if (sh[k][threadIdx.x + half] > sh[k][threadIdx.x])
                    sh[k][threadIdx.x] = sh[k][threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part[blockIdx.x] = sh[0][0];
        part[gridDim.x + blockIdx.x] = sh[1][0];
    }
}

/* One block; a maximum does not depend on the order it is taken in. The
 * guard's count is the device's own, so the launch arguments are the same
 * every step (a captured step replays it). Its two values go to the pinned
 * ring, then the count the host waits on; with over > 0 a step that keeps its
 * list with d_max >= over is counted in idx[1]. */
__global__ void gcn_kern_guard_finalize(const gc_f64 *__restrict__ part,
                                        int nblk, gc_f64 *__restrict__ ring,
                                        gc_i64 *__restrict__ idx,
                                        gc_i64 *__restrict__ done,
                                        double over,
                                        gc_f64 *__restrict__ reset)
{
    __shared__ double sh[2][GCN_RED_BLOCK];
    double m = 0.0, ms = 0.0;
    for (int i = threadIdx.x; i < nblk; i += blockDim.x) {
        if (part[i] > m) m = part[i];
        if (part[nblk + i] > ms) ms = part[nblk + i];
    }
    sh[0][threadIdx.x] = m;
    sh[1][threadIdx.x] = ms;
    __syncthreads();
    for (int half = blockDim.x >> 1; half > 0; half >>= 1) {
        if ((int)threadIdx.x < half) {
            for (int k = 0; k < 2; ++k)
                if (sh[k][threadIdx.x + half] > sh[k][threadIdx.x])
                    sh[k][threadIdx.x] = sh[k][threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const gc_i64 n = (*idx)++;
        const double d = sqrt(sh[0][0]);
        volatile gc_f64 *slot = ring + 2 * (n % GC_GUARD_RING);
        slot[0] = d;
        slot[1] = sqrt(sh[1][0]);
        if (over > 0.0 && d >= over) idx[1]++;
        if (reset) reset[0] = reset[1] = 0.0;
        __threadfence_system();
        *(volatile gc_i64 *)done = n + 1;
    }
}

__global__ void gcn_kern_copy3(gc_f64 *__restrict__ dst,
                               const gc_f64 *__restrict__ src, gc_i64 n3)
{
    for (gc_i64 s = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; s < n3;
         s += (gc_i64)gridDim.x * blockDim.x) dst[s] = src[s];
}

/* Every member is translated with its group's representative image, frozen at
 * each rebuild; raw integrator coordinates stay intact. */
__global__ void gcn_kern_force_coord(const gc_f64 *__restrict__ raw,
                                     const gc_i64 *__restrict__ goff,
                                     gc_f64 *__restrict__ view,
                                     gc_f64 *__restrict__ saved_move,
                                     gc_i64 *__restrict__ bad,
                                     gc_i64 ng, gc_i64 na, gc_i64 pitch,
                                     double bx, double by, double bz,
                                     int capture)
{
    for (gc_i64 g = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; g < ng;
         g += (gc_i64)gridDim.x * blockDim.x) {
        const gc_i64 b = goff[g], e = goff[g + 1];
        if (b < 0 || e <= b || e > na) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        const double box[3] = { bx, by, bz };
        double move[3];
        int valid = 1;
        for (int k = 0; k < 3; ++k) {
            if (capture) {
                const double rep = raw[(gc_i64)k * pitch + b];
                const double q = rep / box[k];
                if (!isfinite(q) || fabs(q) > 4503599627370496.0) {
                    valid = 0;
                    break;
                }
                move[k] = box[k] * 0.5 - box[k] * round(q);
                if (isfinite(move[k])) saved_move[3 * g + k] = move[k];
            } else {
                move[k] = saved_move[3 * g + k];
            }
            if (!isfinite(move[k])) { valid = 0; break; }
        }
        if (!valid) {
            atomicAdd((unsigned long long *)bad, 1ull);
            continue;
        }
        for (gc_i64 s = b; s < e; ++s) {
            for (int k = 0; k < 3; ++k) {
                const gc_i64 at = (gc_i64)k * pitch + s;
                const double x = raw[at];
                if (!isfinite(x) || !isfinite(x + move[k])) {
                    atomicAdd((unsigned long long *)bad, 1ull);
                    continue;
                }
                view[at] = x + move[k];
            }
        }
    }
}

/* Host side */

namespace gcn {

gc_status dev_calloc(void **p, gc_i64 bytes)
{
    *p = 0;
    if (bytes <= 0) return GC_OK;
    if (cudaMalloc(p, (size_t)bytes) != cudaSuccess) return GC_E_NOMEM;
    if (cudaMemset(*p, 0, (size_t)bytes) != cudaSuccess) return GC_E_DEVICE;
    return GC_OK;
}

void dev_release(void **p)
{
    if (*p) { cudaFree(*p); *p = 0; }
}

/* Rebuild scratch and the resident canonical-term copies */

/* Every temporary the rebuild needs is a grow-only slot, one per call site,
 * owned by its gcn_device and released in native_state_free; zeroed on the
 * rebuild stream when asked, contents never carried between rebuilds. */
namespace {
enum gcn_scratch_slot {
    GCN_SC_ENDPOS, GCN_SC_KEEP, GCN_SC_PAIR_AT, GCN_SC_SLOT, GCN_SC_TOTALS,
    GCN_SC_REPORT, GCN_SC_CNT_I, GCN_SC_CNT_J, GCN_SC_OFF_I, GCN_SC_OFF_J,
    GCN_SC_HIT, GCN_SC_BITS, GCN_SC_OFF_M, GCN_SC_MKEY, GCN_SC_MIKEY,
    GCN_SC_MVAL, GCN_SC_RM_EP, GCN_SC_RM_EPI, GCN_SC_RM_EX,
    GCN_SC_SCANTMP, GCN_SC_CBOX, GCN_SC_FLAGS, GCN_SC_CATOMS, GCN_SC_INV, GCN_SC_CARRYCHK, GCN_SC_CARRYMAP, GCN_SC_GSORT, GCN_SC_AVIEW, GCN_SC_PLACE, GCN_SC_PORDER, GCN_SC_MIGLIST, GCN_SC_BIG,
    GCN_SC_N
};

/* The device copy of one kind's canonical records (endpoint gids and images,
 * parameter id, source pbc codes, per-term stock image), inputs of the term
 * stage. Uploaded at import; once a
 * product migration derived the terms on the device they are the pool. */
struct gcn_term_copy {
    gc_gid   *ep  = 0;  gc_i64 ep_cap  = 0;
    gc_image *epi = 0;  gc_i64 epi_cap = 0;
    gc_i32   *pid = 0;  gc_i64 pid_cap = 0;
    gc_i32   *pbc = 0;  gc_i64 pbc_cap = 0;
    gc_i32   *img = 0;  gc_i64 img_cap = 0;
    gc_i64 nt = -1;
    const void *h_ep = 0, *h_epi = 0, *h_pid = 0, *h_pbc = 0, *h_img = 0;
    bool valid = false;
};

struct gcn_rebuild_cache {
    void  *sc[GCN_SC_N]  = {};
    gc_i64 cap[GCN_SC_N] = {};
    gc_i64 *pin = 0;            /* pinned readback words (GCN_PIN_N)    */
    gc_i64 sel_words = 0;       /* the selection's bit words: grow-only  */
    gcn_term_copy term[GC_TERM_NKIND];
    gcn_term_copy real_mask;
    /* The execution records the last build left in d->term[] (and the real
     * mask's) describe that build's slots; the next one-rank build with no
     * ghost band may map them through its permutation. */
    bool exec_ok = false;
    gc_i64 exec_nt[GC_TERM_NKIND] = {};
    gc_i64 exec_nm = -1;
    const void *exec_rm = 0;
    gc_i64 exec_nres = -1;
    gc_i64 term_cap[GC_TERM_NKIND][3] = {};
    /* The device pools are canonical (a migration derived them); cleared by
     * every import. */
    bool term_dev = false;
    /* Likewise the zero-bit rows: real_mask is canonical once a product
     * migration edited it; the last selection is kept for
     * native_mask_replace. */
    bool mask_dev = false;
    gcn_term_copy mask_alt;
    gc_i32 *msel = 0; gc_i64 msel_cap = 0;
    gc_i64 *mpos = 0; gc_i64 mpos_cap = 0;
    gc_i64 mnsel = 0;
    void *mq = 0;      gc_i64 mq_cap = 0;
    void *mgather = 0; gc_i64 mgather_cap = 0;
    void *mstage = 0;  gc_i64 mstage_cap = 0;
    double mrate = 0.0;                       /* most rows per query so far */
    std::vector<gc_gid> mpre_q;
    gc_i64 mpre_room = -1;
    /* The replicated topology (native_terms_topology): every rank's
     * imported terms of a kind, ordered by anchor GID (endpoint 0) and then
     * by (rank, pool index), and a CSR by anchor GID (topo_first[k][g] ..
     * topo_first[k][g + 1]).  Parameter ids index the ranks' concatenated
     * payloads, conditioned in topo_real / topo_int until the first
     * derivation (topo_live) replaces the import's tables with them. */
    bool topo_ready = false, topo_live = false;
    gcn_term_copy topo[GC_TERM_NKIND];
    gc_i32 *topo_first[GC_TERM_NKIND] = {};
    gc_i64 topo_first_cap[GC_TERM_NKIND] = {};
    gc_gid topo_ngid = 0;
    std::vector<gc_f64> topo_real[GC_TERM_NKIND];
    std::vector<gc_i32> topo_int[GC_TERM_NKIND];
    gc_i64 topo_stride[GC_TERM_NKIND] = {};
    gc_i64 *tcount = 0; gc_i64 tcount_cap = 0;   /* per kind, per atom    */
    gc_i64 *tpos = 0;   gc_i64 tpos_cap = 0;     /* per kind, per atom    */
    gc_i64 *ttot = 0;   gc_i64 ttot_cap = 0;     /* the kinds' totals     */
    gc_i64 *tpin = 0;                            /* and their readback    */
};

std::unordered_map<const struct gcn_device *, gcn_rebuild_cache> g_rebuild_cache;

gcn_rebuild_cache &rebuild_cache(const struct gcn_device *d)
{
    return g_rebuild_cache[d];
}

gc_status grow(void **p, gc_i64 *cap, gc_i64 bytes)
{
    if (bytes <= *cap && *p) return GC_OK;
    if (*p) { cudaFree(*p); *p = 0; *cap = 0; }
    const gc_i64 want = bytes + bytes / 4 + 256;
    if (cudaMalloc(p, (size_t)want) != cudaSuccess) { *p = 0; return GC_E_NOMEM; }
    *cap = want;
    return GC_OK;
}

void term_copy_release(gcn_term_copy &t)
{
    if (t.ep)  cudaFree(t.ep);
    if (t.epi) cudaFree(t.epi);
    if (t.pid) cudaFree(t.pid);
    if (t.pbc) cudaFree(t.pbc);
    if (t.img) cudaFree(t.img);
    t = gcn_term_copy();
}
}  /* anonymous namespace */

static gc_status scratch(struct gcn_device *d, int slot, gc_i64 bytes,
                         cudaStream_t s, int zero, void **out)
{
    *out = 0;
    if (bytes <= 0) return GC_OK;
    gcn_rebuild_cache &c = rebuild_cache(d);
    gc_status st = grow(&c.sc[slot], &c.cap[slot], bytes);
    if (st != GC_OK) return st;
    if (zero && cudaMemsetAsync(c.sc[slot], 0, (size_t)bytes, s) != cudaSuccess)
        return GC_E_DEVICE;
    *out = c.sc[slot];
    return GC_OK;
}

#define GCN_PIN_N 80
static gc_i64 *rebuild_pinned(struct gcn_device *d)
{
    gcn_rebuild_cache &c = rebuild_cache(d);
    if (!c.pin && cudaHostAlloc((void **)&c.pin, GCN_PIN_N * sizeof(gc_i64),
                                cudaHostAllocDefault) != cudaSuccess)
        c.pin = 0;
    return c.pin;
}

static inline gc_status rb_phase(cudaStream_t s, const char *name)
{
    (void)s;
    return dev_launched(name);
}

static int sm_count()
{
    static int nsm = 0;
    if (nsm <= 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev)
                != cudaSuccess || nsm <= 0)
            nsm = 80;
    }
    return nsm;
}

static unsigned pair_grid(gc_i64 npair)
{
    const gc_i64 g = (gc_i64)sm_count() * 12;
    return (unsigned)(npair < g ? (npair > 0 ? npair : 1) : g);
}

static inline void scratch_drop(void **p) { *p = 0; }

template <typename T>
__global__ void gcn_kern_scan_total(const T *__restrict__ in,
                                    const gc_i64 *__restrict__ out, gc_i64 n,
                                    gc_i64 *__restrict__ total)
{
    *total = out[n - 1] + (gc_i64)in[n - 1];
}

/* Exclusive scan of n elements on s, summed in 64 bits (exact in any order);
 * the total, when asked, lands in *total on the device. */
template <typename T>
static gc_status mscan(struct gcn_device *d, const T *in, gc_i64 *out,
                       gc_i64 n, gc_i64 *total, cudaStream_t s)
{
    if (n <= 0) {
        if (total && cudaMemsetAsync(total, 0, sizeof(gc_i64), s) != cudaSuccess)
            return GC_E_DEVICE;
        return GC_OK;
    }
    if (n > (gc_i64)INT_MAX) return GC_E_CAPACITY;
    gcn_rebuild_cache &c = rebuild_cache(d);
    size_t need = 0;
    if (cub::DeviceScan::ExclusiveScan(nullptr, need, in, out,
                                       cuda::std::plus<gc_i64>(), (gc_i64)0,
                                       (int)n, s) != cudaSuccess)
        return GC_E_DEVICE;
    gc_status st = grow(&c.sc[GCN_SC_SCANTMP], &c.cap[GCN_SC_SCANTMP],
                        (gc_i64)need);
    if (st != GC_OK) return st;
    size_t bytes = (size_t)c.cap[GCN_SC_SCANTMP];
    if (cub::DeviceScan::ExclusiveScan(c.sc[GCN_SC_SCANTMP], bytes, in, out,
                                       cuda::std::plus<gc_i64>(), (gc_i64)0,
                                       (int)n, s) != cudaSuccess)
        return GC_E_DEVICE;
    if (total) gcn_kern_scan_total<T><<<1, 1, 0, s>>>(in, out, n, total);
    return dev_launched("mscan");
}

gc_status native_scan(struct gcn_device *d, const gc_i32 *in, gc_i64 *out,
                      gc_i64 n, gc_i64 *total)
{
    return mscan(d, in, out, n, total, d->stream);
}

gc_status native_scan64(struct gcn_device *d, const gc_i64 *in, gc_i64 *out,
                        gc_i64 n, gc_i64 *total)
{
    return mscan(d, in, out, n, total, d->stream);
}

gc_i64 *native_rebuild_pin(struct gcn_device *d) { return rebuild_pinned(d); }

gc_status native_pack_verdict(const gc_context *ctx,
                              const struct gcn_pack_check *pc)
{
    const gc_i64 *packed = pc->packed, bad = pc->packed[3];
    const gc_i64 *candidate = pc->candidate;
    if (!bad && !packed[2] && packed[0] == candidate[0] &&
        packed[1] == candidate[1])
        return GC_OK;
    std::fprintf(stderr,
        "GPU_Core_Error> phase=rebuild_group_pack rank=%d bad=%lld "
        "packed=%lld,%lld,%lld candidate=%lld,%lld,%lld "
        "groups=%lld owned=%lld\n", (int)ctx->rank,
        (long long)bad, (long long)packed[0], (long long)packed[1],
        (long long)packed[2], (long long)candidate[0],
        (long long)candidate[1], (long long)candidate[2],
        (long long)pc->groups, (long long)pc->owned);
    return GC_E_MISMATCH;
}

static void term_copies_invalidate(const struct gcn_device *d)
{
    auto it = g_rebuild_cache.find(d);
    if (it == g_rebuild_cache.end()) return;
    for (int k = 0; k < GC_TERM_NKIND; ++k) it->second.term[k].valid = false;
    it->second.real_mask.valid = false;
    it->second.exec_ok = false;
    it->second.term_dev = false;
    it->second.mask_dev = false;
    it->second.topo_ready = false;   /* a new import, a new topology */
    it->second.topo_live = false;
}

static void rebuild_cache_release(const struct gcn_device *d)
{
    auto it = g_rebuild_cache.find(d);
    if (it == g_rebuild_cache.end()) return;
    for (int i = 0; i < GCN_SC_N; ++i)
        if (it->second.sc[i]) cudaFree(it->second.sc[i]);
    if (it->second.pin) cudaFreeHost(it->second.pin);
    if (it->second.mstage) cudaFreeHost(it->second.mstage);
    if (it->second.mq) cudaFree(it->second.mq);
    if (it->second.mgather) cudaFree(it->second.mgather);
    if (it->second.msel) cudaFree(it->second.msel);
    if (it->second.mpos) cudaFree(it->second.mpos);
    term_copy_release(it->second.mask_alt);
    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        term_copy_release(it->second.term[k]);
    }
    term_copy_release(it->second.real_mask);
    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        term_copy_release(it->second.topo[k]);
        if (it->second.topo_first[k]) cudaFree(it->second.topo_first[k]);
    }
    if (it->second.tcount) cudaFree(it->second.tcount);
    if (it->second.tpos) cudaFree(it->second.tpos);
    if (it->second.ttot) cudaFree(it->second.ttot);
    if (it->second.tpin) cudaFreeHost(it->second.tpin);
    g_rebuild_cache.erase(it);
}

/* Upload (or keep) the device copy of `n` endpoint records.  Reused only
 * when `reusable` and the host arrays are the same arrays of the same
 * length as at the last upload. */
static gc_status term_copy_sync(gcn_term_copy &t, int reusable, gc_i64 nt,
                                gc_i64 nep, gc_i64 ncodes,
                                const gc_gid *h_ep, const gc_image *h_epi,
                                const gc_i32 *h_pid, const gc_i32 *h_pbc,
                                const gc_i32 *h_img, cudaStream_t s)
{
    if (reusable && t.valid && t.nt == nt && t.h_ep == h_ep &&
        t.h_epi == h_epi && t.h_pid == h_pid && t.h_pbc == h_pbc &&
        t.h_img == h_img)
        return GC_OK;
    t.valid = false;
    gc_status st;
    st = grow((void **)&t.ep, &t.ep_cap, nep * (gc_i64)sizeof(gc_gid));
    if (st != GC_OK) return st;
    st = grow((void **)&t.epi, &t.epi_cap, nep * (gc_i64)sizeof(gc_image));
    if (st != GC_OK) return st;
    if (cudaMemcpyAsync(t.ep, h_ep, (size_t)nep * sizeof(gc_gid),
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        cudaMemcpyAsync(t.epi, h_epi, (size_t)nep * sizeof(gc_image),
                        cudaMemcpyHostToDevice, s) != cudaSuccess)
        return GC_E_DEVICE;
    if (h_pid) {
        st = grow((void **)&t.pid, &t.pid_cap, nt * (gc_i64)sizeof(gc_i32));
        if (st != GC_OK) return st;
        if (cudaMemcpyAsync(t.pid, h_pid, (size_t)nt * sizeof(gc_i32),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
    }
    if (h_pbc && ncodes > 0) {
        st = grow((void **)&t.pbc, &t.pbc_cap,
                  nt * ncodes * (gc_i64)sizeof(gc_i32));
        if (st != GC_OK) return st;
        if (cudaMemcpyAsync(t.pbc, h_pbc, (size_t)(nt * ncodes) * sizeof(gc_i32),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
    }
    if (h_img) {
        st = grow((void **)&t.img, &t.img_cap, nt * (gc_i64)sizeof(gc_i32));
        if (st != GC_OK) return st;
        if (cudaMemcpyAsync(t.img, h_img, (size_t)nt * sizeof(gc_i32),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
    }
    t.nt = nt; t.h_ep = h_ep; t.h_epi = h_epi; t.h_pid = h_pid; t.h_pbc = h_pbc;
    t.h_img = h_img;
    t.valid = true;
    return GC_OK;
}

/* Kind k's device pool from the host pool of `nt` records (host-canonical
 * before any device migration). A record
 * the host pool cannot describe refuses. */
static gc_status term_pool_from_host(const gc_context *ctx,
                                     gcn_rebuild_cache &rc, int k, gc_i64 nt,
                                     int reusable, cudaStream_t s)
{
    const gc_i64 nep = nt * gc_term_arity[k];
    const gc_i32 ncodes = gc_term_pbc_codes[k];
    const gcn::TermPool &tp = ctx->term[k];
    if (nt <= 0) return GC_OK;
    if ((gc_i64)tp.endpoint_image.size() != nep ||
        (gc_i64)tp.endpoint_image_valid.size() != nep ||
        (gc_i64)tp.endpoint.size() < nep ||
        (gc_i64)tp.param_id.size() < nt ||
        (gc_i64)tp.image.size() < nt ||
        (ncodes > 0 && (gc_i64)tp.pbc.size() < nt * ncodes))
        return GC_E_ENDPOINT;
    gcn_term_copy &tc = rc.term[k];
    const int fresh = reusable && tc.valid && tc.nt == nt &&
        tc.h_ep == (const void *)&tp.endpoint[0] &&
        tc.h_epi == (const void *)&tp.endpoint_image[0] &&
        tc.h_pid == (const void *)&tp.param_id[0] &&
        tc.h_pbc == (ncodes > 0 ? (const void *)&tp.pbc[0] : 0) &&
        tc.h_img == (const void *)&tp.image[0];
    if (fresh) return GC_OK;
    for (gc_i64 q = 0; q < nep; ++q)
        if (!tp.endpoint_image_valid[(size_t)q]) return GC_E_ENDPOINT;
    return term_copy_sync(tc, reusable, nt, nep, ncodes, &tp.endpoint[0],
                          &tp.endpoint_image[0], &tp.param_id[0],
                          ncodes > 0 ? &tp.pbc[0] : 0, &tp.image[0], s);
}

gc_status dev_phase(cudaStream_t s, const char *phase)
{
    cudaError_t e = cudaStreamSynchronize(s);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e == cudaSuccess) return GC_OK;
    std::fprintf(stderr, "GPU_Core_Error> phase=%s cuda=%s\n",
                 phase, cudaGetErrorString(e));
    return GC_E_DEVICE;
}

namespace {
thread_local std::vector<std::function<gc_status()> > g_host_work;
thread_local size_t g_host_next = 0;
thread_local gc_status g_host_st = GC_OK;
}

void host_defer(std::function<gc_status()> work)
{
    g_host_work.push_back(std::move(work));
}

gc_status host_run()
{
    while (g_host_next < g_host_work.size()) {
        const gc_status st = g_host_work[g_host_next++]();
        if (g_host_st == GC_OK) g_host_st = st;
    }
    return g_host_st;
}

gc_status host_drain()
{
    const gc_status st = host_run();
    g_host_work.clear();
    g_host_next = 0;
    g_host_st = GC_OK;
    return st;
}

gc_status dev_launched(const char *phase)
{
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return GC_OK;
    std::fprintf(stderr, "GPU_Core_Error> phase=%s cuda=%s\n",
                 phase, cudaGetErrorString(e));
    return GC_E_DEVICE;
}

gc_status dev_sync(cudaStream_t s)
{
    return dev_phase(s, "unnamed");
}

static gc_i64 grid_for(gc_i64 n, gc_i64 block);

static gc_status force_coords_launch(gc_context *ctx, gc_i32 capture_move,
                                     gc_i64 *counter, int clear)
{
    if (!ctx || !ctx->native) return GC_E_ARG;
    struct gcn_device *d = ctx->native;
    if (!d->force_coord || !d->group_force_move ||
        d->num_groups <= 0 || d->num_owned <= 0)
        return GC_E_STATE;
    if (capture_move != 0 && capture_move != 1) return GC_E_ARG;
    if (!capture_move && !d->force_move_valid) return GC_E_STATE;
    for (int k = 0; k < 3; ++k)
        if (!(d->box[k] > 0.0) || !std::isfinite(d->box[k])) return GC_E_ARG;

    if (capture_move) d->force_move_valid = 0;
    if (clear && cudaMemsetAsync(counter, 0, sizeof(gc_i64), d->stream)
            != cudaSuccess) return GC_E_DEVICE;
    gcn_kern_force_coord<<<(unsigned)grid_for(d->num_groups, GCN_BLOCK),
                           GCN_BLOCK, 0, d->stream>>>(
        d->coord, d->group_offset, d->force_coord, d->group_force_move,
        counter,
        d->num_groups, d->num_owned, d->pitch,
        d->box[0], d->box[1], d->box[2], capture_move);
    return dev_launched("force_coordinates");
}

/* The step's view: the same launch, its failures added to the sticky
 * verdict[0] and read at the next synchronising call instead of here. */
gc_status native_force_coordinates_step(gc_context *ctx)
{
    return force_coords_launch(ctx, 0,
                               ctx && ctx->native ? ctx->native->verdict : 0,
                               0);
}

static gc_i64 grid_for(gc_i64 n, gc_i64 block)
{
    /* the grid-stride kernels: enough blocks to fill every SM's thread
       slots (at least GCN_MAX_BLOCKS), never more than the work */
    const gc_i64 cap = std::max<gc_i64>(GCN_MAX_BLOCKS,
                                        (gc_i64)sm_count() * (2048 / block));
    gc_i64 g = (n + block - 1) / block;
    if (g > cap) g = cap;
    if (g < 1) g = 1;
    return g;
}

gc_status native_state_free(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    if (d == 0) return GC_OK;

    native_pme_release(ctx);
    native_nbc_release(d);
    native_step_release(d);
    native_migration_release(ctx);
    native_dist_destroy(ctx);
    rebuild_cache_release(d);

    void **p[] = {
        (void **)&d->coord, (void **)&d->force_coord,
        (void **)&d->coord_ref, (void **)&d->list_ref,
        &d->vel.p, (void **)&d->vel_ref, (void **)&d->vel_half,
        (void **)&d->vel_full, &d->force.p, (void **)&d->force_real,
        (void **)&d->force_bond, (void **)&d->force_recip,
        (void **)&d->force_real_fx, (void **)&d->force_bond_fx,
        (void **)&d->charge, (void **)&d->mass, (void **)&d->inv_mass,
        (void **)&d->cls, (void **)&d->gid, (void **)&d->image,
        (void **)&d->coord2, (void **)&d->coord_ref2, &d->vel2.p,
        (void **)&d->vel_ref2, (void **)&d->vel_half2, (void **)&d->vel_full2,
        (void **)&d->charge2, (void **)&d->mass2, (void **)&d->inv_mass2,
        (void **)&d->cls2, (void **)&d->gid2,
        (void **)&d->owner, (void **)&d->cell_of,
        (void **)&d->resident_slot, (void **)&d->resident_offset,
        (void **)&d->cell, (void **)&d->group_offset,
        (void **)&d->group_member, (void **)&d->group_kind,
        (void **)&d->group_cell, (void **)&d->group_gid,
        (void **)&d->rigid_gid, (void **)&d->rigid_arity,
        (void **)&d->rigid_dist,
        (void **)&d->group_dest_rank, (void **)&d->group_dest_coord,
        (void **)&d->migration_counts,
        (void **)&d->migration_group_offset,
        (void **)&d->migration_atom_offset,
        (void **)&d->migration_pack_totals,
        (void **)&d->migration_peer_counts,
        (void **)&d->migration_peer_group_base,
        (void **)&d->migration_peer_atom_base,
        (void **)&d->migration_group_send,
        (void **)&d->migration_atom_send,
        (void **)&d->group_force_move,
        (void **)&d->group_offset2, (void **)&d->group_member2,
        (void **)&d->group_kind2, (void **)&d->group_cell2,
        (void **)&d->group_gid2,
        (void **)&d->pair, (void **)&d->pair_order, (void **)&d->sel,
        (void **)&d->mask,
        (void **)&d->sort_key_hi, (void **)&d->sort_key_lo,
        (void **)&d->sort_val, (void **)&d->sort_key_hi2,
        (void **)&d->sort_key_lo2, (void **)&d->sort_val2,
        (void **)&d->perm, (void **)&d->scratch3,
        (void **)&d->water_slot, (void **)&d->water_group,
        (void **)&d->hgr_heavy, (void **)&d->hgr_group, (void **)&d->hgr_h,
        (void **)&d->hgr_dist, (void **)&d->hgr_inv_mass_h,
        (void **)&d->hgr_inv_mass_c, (void **)&d->hgr_arity,
        (void **)&d->rigid_tmp, (void **)&d->tick_part,
        (void **)&d->lj12, (void **)&d->lj6, (void **)&d->nb14_lj12,
        (void **)&d->nb14_lj6, (void **)&d->table_ene, (void **)&d->table_grad,
        (void **)&d->table_ecor, (void **)&d->table_decor, (void **)&d->nb_mixed,
        (void **)&d->cmap_coef, (void **)&d->cmap_resolution,
        (void **)&d->posres_key, (void **)&d->posres_val,
        (void **)&d->posres_par,
        (void **)&d->acc_real, (void **)&d->acc_bond, (void **)&d->acc_recip,
        (void **)&d->reduce_partial, (void **)&d->reduce_out,
        (void **)&d->fail_counter
    };
    for (unsigned i = 0; i < sizeof(p) / sizeof(p[0]); ++i) dev_release(p[i]);
    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        dev_release((void **)&d->term[k]);
        dev_release((void **)&d->term_param[k]);
        dev_release((void **)&d->term_param_int[k]);
        dev_release((void **)&d->term_pair[k]);
        d->term_pair_cap[k] = d->term_pair_n[k] = 0;
    }
    dev_release((void **)&d->excl_keep);
    dev_release((void **)&d->excl_keep_wid);
    d->excl_keep_cap = d->excl_keep_wid_cap = d->excl_keep_n = 0;
    d->excl_keep_valid = 0;
    dev_release((void **)&d->verdict);
    if (d->guard_pin) { cudaFreeHost(d->guard_pin); d->guard_pin = 0; }
    if (d->stream)     cudaStreamDestroy(d->stream);
    if (d->real_stream) cudaStreamDestroy(d->real_stream);
    if (d->real_done)   cudaEventDestroy(d->real_done);
    if (d->pme_stream)  cudaStreamDestroy(d->pme_stream);
    if (d->pme_fork)    cudaEventDestroy(d->pme_fork);
    if (d->pme_done)    cudaEventDestroy(d->pme_done);

    delete d;
    ctx->native = 0;
    return GC_OK;
}

gc_status native_state_alloc(gc_context *ctx)
{
    if (ctx->native) native_state_free(ctx);

    /* Stock setup makes CUDA calls whose status it does not read; a failed
     * one leaves the runtime's last error set, which dev_phase would report
     * as a device fault. Take it here. */
    const cudaError_t prior = cudaGetLastError();
    if (prior != cudaSuccess)
        std::fprintf(stderr, "GPU_Core_Note> phase=native_state_alloc "
                     "cleared a CUDA error left by earlier stock code: %s\n",
                     cudaGetErrorString(prior));

    struct gcn_device *d = new gcn_device;
    std::memset(d, 0, sizeof(*d));
    ctx->native = d;

    int prio_low = 0, prio_high = 0;
    if (cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high) != cudaSuccess ||
        cudaStreamCreateWithPriority(&d->stream, cudaStreamDefault,
                                     gcn_stream_priority(prio_low, prio_high, 1))
            != cudaSuccess ||
        cudaStreamCreateWithPriority(&d->real_stream, cudaStreamNonBlocking,
                                     ctx->nproc > 1 ? prio_low
                                     : gcn_stream_priority(prio_low, prio_high, 1))
            != cudaSuccess ||
        cudaStreamCreateWithPriority(&d->pme_stream, cudaStreamNonBlocking,
                                     ctx->nproc > 1 ? prio_high
                                     : gcn_stream_priority(prio_low, prio_high, 1))
            != cudaSuccess)
        return GC_E_DEVICE;
    if (cudaEventCreateWithFlags(&d->real_done, cudaEventDisableTiming)
            != cudaSuccess ||
        cudaEventCreateWithFlags(&d->pme_fork, cudaEventDisableTiming)
            != cudaSuccess ||
        cudaEventCreateWithFlags(&d->pme_done, cudaEventDisableTiming)
            != cudaSuccess) return GC_E_DEVICE;

    d->num_owned    = ctx->num_owned;
    d->num_ghost = ctx->num_ghost;
    for (int k = 0; k < 6; ++k)
        if (!gcn::add_checked(d->num_ghost, ctx->native_outer_recv[k / 2][k % 2],
                              &d->num_ghost))
            return GC_E_OVERFLOW;
    if (!gcn::add_checked(d->num_owned, d->num_ghost, &d->num_resident))
        return GC_E_OVERFLOW;
    /* A quarter over the import's counts: migration and the halo refresh grow
     * them, and a regrow is serial on the rank that needs it. */
    const auto room = [](gc_i64 n) { return n + n / 4 + 128; };
    d->pitch = ((room(d->num_resident) + 127) / 128) * 128;

    /* The global cell key is the one-rank instance of the local box layout,
     * so no caller indexes a dense global cell table. */
    {
        const char *why = 0;
        gc_status ls = layout_native(ctx, &d->layout, &why);
        if (ls != GC_OK) {
            if (ctx->rank == 0 && why)
                std::fprintf(stderr, "GPU_Core_Error> phase=layout reason=%s\n", why);
            return ls;
        }
    }
    d->ncell      = d->layout.ncell_box;
    d->num_groups = ctx->num_groups;
    d->num_members= ctx->num_group_members;
    d->owned_cap  = room(d->num_owned);
    d->group_cap  = room(d->num_groups);
    d->member_cap = room(d->num_members);
    d->resident_cap = d->pitch;
    d->sort_capacity = d->group_cap > d->pitch ? d->group_cap : d->pitch;
    d->sort_cap   = d->sort_capacity;
    const gc_i64 gcap = d->group_cap, mcap = d->member_cap;

    const gc_i64 n  = d->pitch;
    const gc_i64 n3 = 3 * d->pitch;
    gc_status st;

#define GCN_ALLOC(field, bytes) \
    do { st = dev_calloc((void **)&d->field, (bytes)); if (st) return st; } while (0)

    GCN_ALLOC(coord,       n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(force_coord, n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(group_force_move, 3 * gcap * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(coord_ref,   n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(list_ref,    n3 * (gc_i64)sizeof(gc_f64));
    /* sized for doubles in either mode (gcn_vf) */
    GCN_ALLOC(vel.p,       n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel_ref,     n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel_half,    n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel_full,    n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(force.p,     n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(force_real,  n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(force_bond,  n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(force_recip, n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(force_real_fx, n3 * (gc_i64)sizeof(unsigned long long));
    GCN_ALLOC(force_bond_fx, n3 * (gc_i64)sizeof(unsigned long long));
    d->fx_clear_pitch[0] = d->fx_clear_pitch[1] = 0;  /* new: clear whole */
    GCN_ALLOC(scratch3,    n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(charge,      n  * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(mass,        n  * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(inv_mass,    n  * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(cls,         n  * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(gid,         n  * (gc_i64)sizeof(gc_gid));
    GCN_ALLOC(coord2,      n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(coord_ref2,  n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel2.p,      n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel_ref2,    n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel_half2,   n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(vel_full2,   n3 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(charge2,     n  * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(mass2,       n  * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(inv_mass2,   n  * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(cls2,        n  * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(gid2,        n  * (gc_i64)sizeof(gc_gid));
    GCN_ALLOC(image,       n  * (gc_i64)sizeof(gc_image));
    GCN_ALLOC(owner,       n  * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(cell_of,     n  * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(resident_slot, d->resident_cap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(resident_offset, (d->ncell + 1) * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(perm,        n  * (gc_i64)sizeof(gc_i32));

    GCN_ALLOC(cell, d->ncell * (gc_i64)sizeof(struct gcn_cell));
    GCN_ALLOC(group_offset, (gcap + 1) * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(group_member, mcap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(group_kind,   gcap * (gc_i64)sizeof(gc_u8));
    GCN_ALLOC(group_cell,   gcap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(group_gid,    gcap * (gc_i64)sizeof(gc_gid));
    GCN_ALLOC(group_dest_rank, gcap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(group_dest_coord, 3 * gcap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(migration_counts, 3 * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_group_offset,
              gcap * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_atom_offset,
              gcap * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_pack_totals, 3 * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_peer_counts,
              3 * (gc_i64)ctx->nproc * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_peer_group_base,
              (gc_i64)ctx->nproc * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_peer_atom_base,
              (gc_i64)ctx->nproc * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(migration_group_send,
              gcap * (gc_i64)sizeof(gcn_group_migration_wire));
    GCN_ALLOC(migration_atom_send,
              d->owned_cap * (gc_i64)sizeof(gcn_atom_migration_wire));
    GCN_ALLOC(group_offset2,(gcap + 1) * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(group_member2, mcap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(group_kind2,   gcap * (gc_i64)sizeof(gc_u8));
    GCN_ALLOC(group_cell2,   gcap * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(group_gid2,    gcap * (gc_i64)sizeof(gc_gid));

    GCN_ALLOC(sort_key_hi,  d->sort_capacity * (gc_i64)sizeof(gc_u64));
    GCN_ALLOC(sort_key_lo,  d->sort_capacity * (gc_i64)sizeof(gc_u64));
    GCN_ALLOC(sort_val,     d->sort_capacity * (gc_i64)sizeof(gc_i32));
    GCN_ALLOC(sort_key_hi2, d->sort_capacity * (gc_i64)sizeof(gc_u64));
    GCN_ALLOC(sort_key_lo2, d->sort_capacity * (gc_i64)sizeof(gc_u64));
    GCN_ALLOC(sort_val2,    d->sort_capacity * (gc_i64)sizeof(gc_i32));

    GCN_ALLOC(acc_real,  2 * GCN_RE_NSLOT * (gc_i64)sizeof(unsigned long long));
    GCN_ALLOC(acc_bond,  2 * GCN_BE_NSLOT * (gc_i64)sizeof(unsigned long long));
    GCN_ALLOC(acc_recip, GCN_PE_NSLOT * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(reduce_partial, 16 * (gc_i64)GCN_MAX_BLOCKS * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(reduce_out, 16 * (gc_i64)sizeof(gc_f64));
    GCN_ALLOC(fail_counter, 4 * (gc_i64)sizeof(gc_i64));
    GCN_ALLOC(verdict, GCN_VERDICT_W * (gc_i64)sizeof(gc_i64));

#undef GCN_ALLOC

    if (cudaHostAlloc((void **)&d->guard_pin,
                      (2 * GC_GUARD_RING + 1) * sizeof(gc_f64),
                      cudaHostAllocDefault) != cudaSuccess)
        return GC_E_NOMEM;
    d->guard_done = (gc_i64 *)(d->guard_pin + 2 * GC_GUARD_RING);
    *(volatile gc_i64 *)d->guard_done = 0;
    d->guard_seq = d->guard_read = 0;
    if (d->guard_idx &&
        cudaMemset(d->guard_idx, 0, 2 * sizeof(gc_i64)) != cudaSuccess)
        return GC_E_DEVICE;
    d->self_epoch = -1;

    return GC_OK;
}

/* Rows [from, to) of kind k's host payload in the kernels' layout: `real`
 * (*stride doubles a row) and `ints` (one a row), each empty when the kind
 * has none. */
static void condition_params(const gcn::TermPool &tp, int k, gc_i64 from,
                             gc_i64 to, std::vector<gc_f64> &real,
                             std::vector<gc_i32> &ints, gc_i64 *stride)
{
    const gc_i64 ps = tp.param_stride, n = to - from;
    real.clear();
    ints.clear();
    if (k == GC_TERM_DIHEDRAL || k == GC_TERM_IMPROPER) {
        *stride = 3;
        real.resize((size_t)(n * 3), 0.0);
        ints.resize((size_t)n, 0);
        for (gc_i64 r = 0; r < n; ++r) {
            const gc_i64 i = from + r;
            double fc    = tp.param_payload[(size_t)(i * ps + 0)];
            double phase = (ps > 1) ? tp.param_payload[(size_t)(i*ps + 1)]
                                    : 0.0;
            double nper  = (ps > 2) ? tp.param_payload[(size_t)(i*ps + 2)]
                                    : 0.0;
            real[(size_t)(r * 3 + 0)] = fc;
            real[(size_t)(r * 3 + 1)] = cos(phase);
            real[(size_t)(r * 3 + 2)] = sin(phase);
            ints[(size_t)r] = (gc_i32)(nper + (nper < 0.0 ? -0.5 : 0.5));
        }
    } else if (k == GC_TERM_CMAP) {
        *stride = 0;
        ints.resize((size_t)n, 0);
        for (gc_i64 r = 0; r < n; ++r) {
            double t0 = tp.param_payload[(size_t)((from + r) * ps)];
            ints[(size_t)r] = (gc_i32)(t0 + 0.5);
        }
    } else {
        *stride = ps;
        real.assign(tp.param_payload.begin() + (long)(from * ps),
                    tp.param_payload.begin() + (long)(to * ps));
    }
}

/* A device buffer holding at least `bytes`, its first `keep` bytes kept. */
static gc_status grow_keep(void **p, gc_i64 *cap, gc_i64 bytes, gc_i64 keep,
                           cudaStream_t s)
{
    if (bytes <= *cap && *p) return GC_OK;
    const gc_i64 want = bytes + bytes / 4 + 256;
    void *q = 0;
    if (cudaMalloc(&q, (size_t)want) != cudaSuccess) return GC_E_NOMEM;
    if (keep > 0 && *p &&
        (cudaMemcpyAsync(q, *p, (size_t)keep, cudaMemcpyDeviceToDevice, s) !=
             cudaSuccess ||
         cudaStreamSynchronize(s) != cudaSuccess)) {
        cudaFree(q);
        return GC_E_DEVICE;
    }
    if (*p) cudaFree(*p);
    *p = q;
    *cap = want;
    return GC_OK;
}

/* One canonical term pool into the device inventory at import: the count, the
 * execution array the term stage writes, and the imported parameters
 * conditioned into the kernels' layout (GENESIS's fields in its order, a
 * deduplicated payload of nreal doubles then nint more). A dihedral's phase
 * becomes its cosine and sine and its periodicity an integer rotation count,
 * as calculate_dihedral_2's caller uses; other kinds are copied verbatim.
 * native_migration_terms_follow carries a migration's changes. */
static gc_status publish_term_pool(const gc_context *ctx,
                                   struct gcn_device *d, int k)
{
    const gcn::TermPool &tp = ctx->term[k];
    gcn_rebuild_cache &rc = rebuild_cache(d);
    gc_i64 *cap = rc.term_cap[k];
    void **buf[3] = { (void **)&d->term[k], (void **)&d->term_param[k],
                      (void **)&d->term_param_int[k] };
    const auto hold = [&](int b, gc_i64 bytes) -> gc_status {
        if (bytes > cap[b]) {
            dev_release(buf[b]);
            cap[b] = 0;
            gc_status st = dev_calloc(buf[b], bytes + bytes / 4);
            if (st) return st;
            cap[b] = bytes + bytes / 4;
            return GC_OK;
        }
        return cudaMemset(*buf[b], 0, (size_t)bytes) == cudaSuccess
               ? GC_OK : GC_E_DEVICE;
    };
    const auto drop = [&](int b) { dev_release(buf[b]); cap[b] = 0; };
    d->term_count[k] = tp.count;
    d->term_param_stride[k] = 0;
    if (tp.count <= 0) { drop(0); drop(1); drop(2); return GC_OK; }
    gc_status st;
    st = hold(0, tp.count * (gc_i64)sizeof(struct gcn_term_exec));
    if (st) return st;
    if (tp.param_count <= 0 || tp.param_stride <= 0) {
        drop(1); drop(2);
        return GC_OK;
    }

    std::vector<gc_f64> real;
    std::vector<gc_i32> ints;
    gc_i64 stride = 0;
    condition_params(tp, k, 0, tp.param_count, real, ints, &stride);
    d->term_param_stride[k] = stride;
    if (real.empty()) drop(1);
    if (ints.empty()) drop(2);
    if (!real.empty()) {
        st = hold(1, (gc_i64)real.size() * (gc_i64)sizeof(gc_f64));
        if (st) return st;
        if (cudaMemcpy(d->term_param[k], &real[0],
                       real.size() * sizeof(gc_f64),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
    }
    if (!ints.empty()) {
        st = hold(2, (gc_i64)ints.size() * (gc_i64)sizeof(gc_i32));
        if (st) return st;
        if (cudaMemcpy(d->term_param_int[k], &ints[0],
                       ints.size() * sizeof(gc_i32),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
    }
    return GC_OK;
}

/* sel[t] = 1 when one of record t's first `nend` endpoints (a term's
 * anchor; either end of a zero-bit row) is in the sorted GID list q. */
__global__ void gcn_kern_term_select(const gc_gid *__restrict__ ep, int arity,
                                     int nend, gc_i64 nt,
                                     const gc_gid *__restrict__ q, gc_i64 nq,
                                     gc_i32 *__restrict__ sel)
{
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; t < nt;
         t += (gc_i64)gridDim.x * blockDim.x) {
        int hit = 0;
        for (int e = 0; e < nend && !hit; ++e) {
            const gc_gid x = ep[t * arity + e];
            gc_i64 lo = 0, hi = nq;
            while (lo < hi) {
                const gc_i64 mid = (lo + hi) / 2;
                if (q[mid] < x) lo = mid + 1; else hi = mid;
            }
            hit = lo < nq && q[lo] == x;
        }
        sel[t] = hit;
    }
}

__global__ void gcn_kern_term_gather(const gc_i32 *__restrict__ sel,
                                     const gc_i64 *__restrict__ pos, gc_i64 nt,
                                     gc_i64 cap, int arity, int ncodes,
                                     const gc_gid *__restrict__ ep,
                                     const gc_image *__restrict__ epi,
                                     const gc_i32 *__restrict__ img,
                                     const gc_i32 *__restrict__ pid,
                                     const gc_i32 *__restrict__ pbc,
                                     gc_i64 *__restrict__ o_idx,
                                     gc_gid *__restrict__ o_ep,
                                     gc_image *__restrict__ o_epi,
                                     gc_i32 *__restrict__ o_img,
                                     gc_i32 *__restrict__ o_pid,
                                     gc_i32 *__restrict__ o_pbc)
{
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; t < nt;
         t += (gc_i64)gridDim.x * blockDim.x) {
        if (!sel[t]) continue;
        const gc_i64 i = pos[t];
        if (i >= cap) continue;
        if (o_idx) o_idx[i] = t;
        for (int e = 0; e < arity; ++e) o_ep[i * arity + e] = ep[t * arity + e];
        if (epi)
            for (int e = 0; e < arity; ++e) o_epi[i * arity + e] = epi[t * arity + e];
        if (img) o_img[i] = img[t];
        if (pid) o_pid[i] = pid[t];
        for (int c = 0; c < ncodes; ++c) o_pbc[i * ncodes + c] = pbc[t * ncodes + c];
    }
}

__global__ void gcn_kern_term_compact(const gc_i32 *__restrict__ sel,
                                      const gc_i64 *__restrict__ pos, gc_i64 nt,
                                      int arity, int ncodes,
                                      const gc_gid *__restrict__ ep,
                                      const gc_image *__restrict__ epi,
                                      const gc_i32 *__restrict__ pid,
                                      const gc_i32 *__restrict__ pbc,
                                      const gc_i32 *__restrict__ img,
                                      gc_gid *__restrict__ o_ep,
                                      gc_image *__restrict__ o_epi,
                                      gc_i32 *__restrict__ o_pid,
                                      gc_i32 *__restrict__ o_pbc,
                                      gc_i32 *__restrict__ o_img)
{
    for (gc_i64 t = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; t < nt;
         t += (gc_i64)gridDim.x * blockDim.x) {
        if (sel[t]) continue;
        const gc_i64 w = t - pos[t];
        for (int e = 0; e < arity; ++e) {
            o_ep[w * arity + e] = ep[t * arity + e];
            o_epi[w * arity + e] = epi[t * arity + e];
        }
        if (pid) o_pid[w] = pid[t];
        if (img) o_img[w] = img[t];
        for (int c = 0; c < ncodes; ++c) o_pbc[w * ncodes + c] = pbc[t * ncodes + c];
    }
}

/* Every kind's pool as one index space: record t of the space is record t -
 * base[k] of kind k, base[k] <= t < base[k + 1]. The migration selects,
 * gathers and compacts all kinds with one launch each. */
/* ---- the replicated topology and the per-rank derivation ------------- */

/* Kind k's replicated topology: the arrays in anchor order and the CSR by
 * anchor GID, uploaded once; the ranks' concatenated parameter payload
 * conditioned into the kernels' layout for the first derivation.  Called
 * with the same data on every rank (the caller gathered it). */
gc_status native_terms_topology(gc_context *ctx, const gcn_term_topo_in *in,
                                gc_gid ngid)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d || !in || ngid < 0 || ngid >= (gc_gid)INT_MAX)
        return GC_E_ARG;
    gcn_rebuild_cache &rc = rebuild_cache(d);
    rc.topo_ready = false;
    rc.topo_live = false;
    cudaStream_t s = d->stream;
    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        const gcn_term_topo_in &t = in[k];
        const gc_i64 n = t.n, ar = gc_term_arity[k], nc = gc_term_pbc_codes[k];
        if (n < 0 || n >= (gc_i64)INT_MAX) return GC_E_CAPACITY;
        rc.topo[k].valid = false;
        gc_status st = term_copy_sync(rc.topo[k], 0, n > 0 ? n : 0, n * ar, nc,
                                      t.ep, t.epi, t.pid, t.pbc, t.img, s);
        if (st == GC_OK)
            st = grow((void **)&rc.topo_first[k], &rc.topo_first_cap[k],
                      (ngid + 1) * (gc_i64)sizeof(gc_i32));
        if (st == GC_OK &&
            cudaMemcpyAsync(rc.topo_first[k], t.first,
                            (size_t)(ngid + 1) * sizeof(gc_i32),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            st = GC_E_DEVICE;
        if (st != GC_OK) return st;
        gcn::TermPool pay;
        pay.param_stride = t.param_stride;
        pay.param_count = t.param_count;
        if (t.param_count > 0 && t.param_stride > 0)
            pay.param_payload.assign(t.payload,
                                     t.payload + t.param_count * t.param_stride);
        rc.topo_stride[k] = 0;
        rc.topo_real[k].clear();
        rc.topo_int[k].clear();
        if (t.param_count > 0 && t.param_stride > 0)
            condition_params(pay, k, 0, t.param_count, rc.topo_real[k],
                             rc.topo_int[k], &rc.topo_stride[k]);
    }
    rc.topo_ngid = ngid;
    const gc_status st = dev_phase(s, "term_topology");
    rc.topo_ready = st == GC_OK;
    return st;
}

void native_terms_topology_reset(gc_context *ctx)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!d) return;
    gcn_rebuild_cache &rc = rebuild_cache(d);
    if (!rc.topo_live) rc.topo_ready = false;
}

int native_terms_topology_ready(const gc_context *ctx)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!d) return 0;
    auto it = g_rebuild_cache.find(d);
    return it != g_rebuild_cache.end() && it->second.topo_ready;
}

struct gcn_term_firsts { const gc_i32 *p[GC_TERM_NKIND]; };

/* count[k * n + i]: the kind-k terms anchored on owned GID i, every kind in
 * one pass, so one scan positions them all. */
__global__ void gcn_kern_term_count(const gc_gid *__restrict__ owned, gc_i64 n,
                                    gcn_term_firsts first, gc_gid ngid,
                                    gc_i64 *__restrict__ count)
{
    for (gc_i64 x = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
         x < GC_TERM_NKIND * n; x += (gc_i64)gridDim.x * blockDim.x) {
        const int k = (int)(x / n);
        const gc_gid g = owned[x - k * n];
        const gc_i32 *f = first.p[k];
        count[x] = (f && g >= 0 && g < ngid) ? (gc_i64)(f[g + 1] - f[g]) : 0;
    }
}

/* Kind k's total from the scan over every kind: the start of the next
 * kind's segment (the grand total after the last) less the start of its own. */
__global__ void gcn_kern_term_totals(const gc_i64 *__restrict__ pos, gc_i64 n,
                                     gc_i64 *__restrict__ tot)
{
    const int k = threadIdx.x;
    if (k < GC_TERM_NKIND)
        tot[k] = (k + 1 < GC_TERM_NKIND ? pos[(k + 1) * n] : tot[GC_TERM_NKIND]) - pos[k * n];
}

__global__ void gcn_kern_term_derive(const gc_gid *__restrict__ owned, gc_i64 n,
                                     const gc_i32 *__restrict__ first,
                                     gc_gid ngid, const gc_i64 *__restrict__ pos,
                                     int arity, int ncodes,
                                     const gc_gid *__restrict__ ep,
                                     const gc_image *__restrict__ epi,
                                     const gc_i32 *__restrict__ img,
                                     const gc_i32 *__restrict__ pid,
                                     const gc_i32 *__restrict__ pbc,
                                     gc_gid *__restrict__ o_ep,
                                     gc_image *__restrict__ o_epi,
                                     gc_i32 *__restrict__ o_img,
                                     gc_i32 *__restrict__ o_pid,
                                     gc_i32 *__restrict__ o_pbc)
{
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (gc_i64)gridDim.x * blockDim.x) {
        const gc_gid g = owned[i];
        if (g < 0 || g >= ngid) continue;
        gc_i64 o = pos[i] - pos[0];   /* pos: this kind's segment of the joint scan */
        for (gc_i64 t = first[g]; t < first[g + 1]; ++t, ++o) {
            for (int e = 0; e < arity; ++e) {
                o_ep[o * arity + e] = ep[t * arity + e];
                o_epi[o * arity + e] = epi[t * arity + e];
            }
            o_img[o] = img[t];
            o_pid[o] = pid[t];
            for (int c = 0; c < ncodes; ++c) o_pbc[o * ncodes + c] = pbc[t * ncodes + c];
        }
    }
}

/* Every kind's pool of this rank: the topology's terms anchored on its n
 * owned GIDs (`owned`, device, sorted), in owned order then topology order. A
 * term is evaluated on the rank that owns its anchor (first endpoint); its
 * forces on ghost endpoints return to their owners through the halo. In FP64
 * those returns are fixed-point words, independent of where a term ran; under
 * nonbond_precision = MIXED they are rounded sums, so placement sets their
 * rounding. The first derivation also replaces the import's parameter tables
 * with the topology's. */
gc_status native_terms_derive(gc_context *ctx, const gc_gid *owned, gc_i64 n)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d || n < 0 || (n > 0 && !owned)) return GC_E_ARG;
    gcn_rebuild_cache &rc = rebuild_cache(d);
    if (!rc.topo_ready) return GC_E_STATE;
    cudaStream_t s = d->stream;
    const gc_i64 N = n > 0 ? n : 1;
    gc_status st = grow((void **)&rc.tcount, &rc.tcount_cap,
                        GC_TERM_NKIND * N * (gc_i64)sizeof(gc_i64));
    if (st == GC_OK)
        st = grow((void **)&rc.tpos, &rc.tpos_cap,
                  GC_TERM_NKIND * N * (gc_i64)sizeof(gc_i64));
    if (st == GC_OK)
        st = grow((void **)&rc.ttot, &rc.ttot_cap,
                  (GC_TERM_NKIND + 1) * (gc_i64)sizeof(gc_i64));
    if (st == GC_OK && !rc.tpin &&
        cudaHostAlloc((void **)&rc.tpin, GC_TERM_NKIND * sizeof(gc_i64),
                      cudaHostAllocDefault) != cudaSuccess) {
        rc.tpin = 0;
        st = GC_E_NOMEM;
    }
    if (st != GC_OK) return st;
    if (n > 0) {
        gcn_term_firsts first;
        for (int k = 0; k < GC_TERM_NKIND; ++k)
            first.p[k] = rc.topo[k].nt > 0 ? rc.topo_first[k] : 0;
        gcn_kern_term_count<<<(unsigned)grid_for(GC_TERM_NKIND * n, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
            owned, n, first, rc.topo_ngid, rc.tcount);
        st = mscan(d, (const gc_i64 *)rc.tcount, rc.tpos, GC_TERM_NKIND * n,
                   rc.ttot + GC_TERM_NKIND, s);
        if (st == GC_OK)
            gcn_kern_term_totals<<<1, 32, 0, s>>>(rc.tpos, n, rc.ttot);
    } else if (cudaMemsetAsync(rc.ttot, 0, GC_TERM_NKIND * sizeof(gc_i64), s) != cudaSuccess) {
        st = GC_E_DEVICE;
    }
    if (st == GC_OK &&
        cudaMemcpyAsync(rc.tpin, rc.ttot, GC_TERM_NKIND * sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
        st = GC_E_DEVICE;
    if (st == GC_OK) st = dev_phase(s, "term_derive_count");
    for (int k = 0; k < GC_TERM_NKIND && st == GC_OK; ++k) {
        const gc_i64 nt = rc.tpin[k], ar = gc_term_arity[k], nc = gc_term_pbc_codes[k];
        if (nt < 0 || nt > rc.topo[k].nt) return GC_E_MISMATCH;
        gcn_term_copy &t = rc.term[k];
        const gc_i64 m = nt > 0 ? nt : 1;
        st = grow((void **)&t.ep, &t.ep_cap, m * ar * (gc_i64)sizeof(gc_gid));
        if (st == GC_OK) st = grow((void **)&t.epi, &t.epi_cap, m * ar * (gc_i64)sizeof(gc_image));
        if (st == GC_OK) st = grow((void **)&t.pid, &t.pid_cap, m * (gc_i64)sizeof(gc_i32));
        if (st == GC_OK) st = grow((void **)&t.img, &t.img_cap, m * (gc_i64)sizeof(gc_i32));
        if (st == GC_OK && nc > 0)
            st = grow((void **)&t.pbc, &t.pbc_cap, m * nc * (gc_i64)sizeof(gc_i32));
        if (st != GC_OK) break;
        if (nt > 0) {
            const gcn_term_copy &g = rc.topo[k];
            gcn_kern_term_derive<<<(unsigned)grid_for(n, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
                owned, n, rc.topo_first[k], rc.topo_ngid, rc.tpos + k * N,
                (int)ar, (int)nc, g.ep, g.epi, g.img, g.pid, g.pbc,
                t.ep, t.epi, t.img, t.pid, t.pbc);
            st = dev_launched("term_derive");
        }
        t.nt = nt;
        t.valid = true;
        t.h_ep = t.h_epi = t.h_pid = t.h_pbc = t.h_img = 0;
        d->term_count[k] = nt;
        gc_i64 *cap = rc.term_cap[k];
        const gc_i64 ex = nt * (gc_i64)sizeof(struct gcn_term_exec);
        if (st == GC_OK && ex > cap[0]) {
            dev_release((void **)&d->term[k]);
            cap[0] = 0;
            st = dev_calloc((void **)&d->term[k], ex + ex / 4);
            if (st == GC_OK) cap[0] = ex + ex / 4;
        }
        if (st == GC_OK && !rc.topo_live) {
            const std::vector<gc_f64> &real = rc.topo_real[k];
            const std::vector<gc_i32> &ints = rc.topo_int[k];
            if (!real.empty()) {
                st = grow_keep((void **)&d->term_param[k], &cap[1],
                               (gc_i64)(real.size() * sizeof(gc_f64)), 0, s);
                if (st == GC_OK &&
                    cudaMemcpy(d->term_param[k], real.data(), real.size() * sizeof(gc_f64),
                               cudaMemcpyHostToDevice) != cudaSuccess)
                    st = GC_E_DEVICE;
            }
            if (st == GC_OK && !ints.empty()) {
                st = grow_keep((void **)&d->term_param_int[k], &cap[2],
                               (gc_i64)(ints.size() * sizeof(gc_i32)), 0, s);
                if (st == GC_OK &&
                    cudaMemcpy(d->term_param_int[k], ints.data(), ints.size() * sizeof(gc_i32),
                               cudaMemcpyHostToDevice) != cudaSuccess)
                    st = GC_E_DEVICE;
            }
            d->term_param_stride[k] = rc.topo_stride[k];
        }
    }
    if (st != GC_OK) return st;
    if (!rc.topo_live) {
        rc.topo_live = true;
        for (int k = 0; k < GC_TERM_NKIND; ++k) {
            std::vector<gc_f64>().swap(rc.topo_real[k]);
            std::vector<gc_i32>().swap(rc.topo_int[k]);
        }
    }
    rc.real_mask.valid = false;
    rc.exec_ok = false;
    rc.term_dev = true;
    return rb_phase(s, "term_derive");
}

/* The device zero-bit rows are the list when a product migration edited them,
 * or when the host list is normalized and unique (`host_unique`) and the last
 * build copied it. */
int native_mask_device(const gc_context *ctx, int host_unique)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!d) return 0;
    const gcn_rebuild_cache &rc = rebuild_cache(d);
    return rc.mask_dev ||
           (host_unique && rc.real_mask.valid &&
            rc.real_mask.nt == (gc_i64)(ctx->real_mask_gid.size() / 2));
}

__global__ void gcn_kern_mask_member(const gc_gid *__restrict__ ep, gc_i64 room,
                                     const gc_i64 *__restrict__ count,
                                     const gc_gid *__restrict__ tab, gc_i64 ntab,
                                     const gc_gid *__restrict__ extra,
                                     gc_i64 nextra, gc_u8 *__restrict__ flag)
{
    const gc_i64 k = *count < room ? *count : room;
    for (gc_i64 i = (gc_i64)blockIdx.x * blockDim.x + threadIdx.x;
         i < 2 * room + nextra; i += (gc_i64)gridDim.x * blockDim.x) {
        if (i < 2 * room && i / 2 >= k) { flag[i] = 0; continue; }
        const gc_gid x = i < 2 * room ? ep[i] : extra[i - 2 * room];
        gc_i64 lo = 0, hi = ntab;
        while (lo < hi) {
            const gc_i64 mid = (lo + hi) / 2;
            if (tab[mid] < x) lo = mid + 1; else hi = mid;
        }
        flag[i] = lo < ntab && tab[lo] == x;
    }
}

/* The device rows naming one of the nq sorted GIDs q, in list order, read
 * back (pinned) as (gid a, image a, gid b, image b); the selection is kept
 * for native_mask_replace.  With `member`, the rows' endpoints and its
 * extra GIDs are looked up in its table in the same pass. */
static gc_status mask_gather_queue(gcn_rebuild_cache &rc, struct gcn_device *d,
                                   gc_i64 nq, const gcn_mask_member *member,
                                   gc_i64 room);

/* The selection's launches for query q (and member's extra GIDs): the
 * selection, its count into pinned word GCN_PIN_N - 8, and the first
 * pass's gather and readback at `room` rows; no synchronisation. */
static gc_status mask_select_queue(gcn_rebuild_cache &rc, struct gcn_device *d,
                                   const gc_gid *q, gc_i64 nq,
                                   const gcn_mask_member *member, gc_i64 room)
{
    cudaStream_t s = d->stream;
    const gcn_term_copy &m = rc.real_mask;
    const gc_i64 nm = m.nt > 0 ? m.nt : 0;
    const gc_i64 nx = member && member->table ? member->nextra : 0;
    gc_status st = grow(&rc.mq, &rc.mq_cap, (nq + nx) * (gc_i64)sizeof(gc_gid));
    if (st == GC_OK) st = grow((void **)&rc.msel, &rc.msel_cap, nm * (gc_i64)sizeof(gc_i32));
    if (st == GC_OK) st = grow((void **)&rc.mpos, &rc.mpos_cap, (nm + 1) * (gc_i64)sizeof(gc_i64));
    if (st != GC_OK) return st;
    gc_i64 *pin = rebuild_pinned(d);
    if (!pin) return GC_E_NOMEM;
    if (cudaMemcpyAsync(rc.mq, q, (size_t)nq * sizeof(gc_gid),
                        cudaMemcpyHostToDevice, s) != cudaSuccess ||
        (nx > 0 && cudaMemcpyAsync((gc_gid *)rc.mq + nq, member->extra,
                                   (size_t)nx * sizeof(gc_gid),
                                   cudaMemcpyHostToDevice, s) != cudaSuccess))
        return GC_E_DEVICE;
    gcn_kern_term_select<<<(unsigned)grid_for(nm, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        m.ep, 2, 2, nm, (const gc_gid *)rc.mq, nq, rc.msel);
    st = mscan(d, (const gc_i32 *)rc.msel, rc.mpos, nm, rc.mpos + nm, s);
    if (st != GC_OK) return st;
    if (cudaMemcpyAsync(pin + GCN_PIN_N - 8, rc.mpos + nm, sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return GC_E_DEVICE;
    return mask_gather_queue(rc, d, nq, member, room);
}

/* One pass's gather of the selection into `room` rows (and the member
 * lookup), and its readback into the stage; no synchronisation. */
static gc_status mask_gather_queue(gcn_rebuild_cache &rc, struct gcn_device *d,
                                   gc_i64 nq, const gcn_mask_member *member,
                                   gc_i64 room)
{
    cudaStream_t s = d->stream;
    const gcn_term_copy &m = rc.real_mask;
    const gc_i64 nm = m.nt > 0 ? m.nt : 0;
    const gc_i64 nx = member && member->table ? member->nextra : 0;
    const int flags = member && member->table;
    const gc_i64 fb = room * (gc_i64)(sizeof(gcn_mask_row) + sizeof(gc_i64));
    const gc_i64 bytes = fb + (flags ? ((2 * room + nx + 7) & ~(gc_i64)7) : 0);
    gc_status st = grow(&rc.mgather, &rc.mgather_cap, bytes);
    if (st != GC_OK) return st;
    if (bytes > rc.mstage_cap) {
        if (rc.mstage) cudaFreeHost(rc.mstage);
        rc.mstage = 0;
        rc.mstage_cap = 0;
        if (cudaHostAlloc(&rc.mstage, (size_t)(bytes + bytes / 4),
                          cudaHostAllocDefault) != cudaSuccess) {
            rc.mstage = 0;
            return GC_E_NOMEM;
        }
        rc.mstage_cap = bytes + bytes / 4;
    }
    gc_gid *gg = (gc_gid *)rc.mgather;
    gc_image *gi = (gc_image *)(gg + 2 * room);
    gc_i64 *gx = (gc_i64 *)(gi + 2 * room);
    gcn_kern_term_gather<<<(unsigned)grid_for(nm, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        rc.msel, rc.mpos, nm, room, 2, 0, m.ep, m.epi, (const gc_i32 *)0,
        (const gc_i32 *)0, (const gc_i32 *)0, gx, gg, gi,
        (gc_i32 *)0, (gc_i32 *)0, (gc_i32 *)0);
    if (flags)
        gcn_kern_mask_member<<<(unsigned)grid_for(2 * room + nx, GCN_BLOCK),
                               GCN_BLOCK, 0, s>>>(
            gg, room, rc.mpos + nm, member->table, member->ntable,
            (const gc_gid *)rc.mq + nq, nx, (gc_u8 *)rc.mgather + fb);
    if (cudaMemcpyAsync(rc.mstage, rc.mgather, (size_t)bytes,
                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return GC_E_DEVICE;
    return GC_OK;
}

static gc_i64 mask_room(const gcn_rebuild_cache &rc, gc_i64 nm, gc_i64 nq)
{
    return std::min(nm, (gc_i64)(rc.mrate * 1.25 * (double)nq) + 32);
}

/* native_mask_select's first pass for the sorted GIDs q, queued so that it
 * lands with the caller's next synchronisation; a native_mask_select of the
 * same query without a member then reads it. */
gc_status native_mask_select_prefetch(gc_context *ctx, const gc_gid *q,
                                      gc_i64 nq)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d || nq < 0 || (nq > 0 && !q)) return GC_E_ARG;
    gcn_rebuild_cache &rc = rebuild_cache(d);
    rc.mpre_room = -1;
    const gc_i64 nm = rc.real_mask.nt > 0 ? rc.real_mask.nt : 0;
    if (nm == 0 || nq == 0) return GC_OK;
    const gc_i64 room = mask_room(rc, nm, nq);
    const gc_status st = mask_select_queue(rc, d, q, nq, 0, room);
    if (st != GC_OK) return st;
    rc.mpre_q.assign(q, q + nq);
    rc.mpre_room = room;
    return dev_launched("mask_select_prefetch");
}

gc_status native_mask_select(gc_context *ctx, const gc_gid *q, gc_i64 nq,
                             const gcn_mask_row **rows, const gc_i64 **index,
                             gc_i64 *n, gcn_mask_member *member)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d || !rows || !index || !n || nq < 0 || (nq > 0 && !q))
        return GC_E_ARG;
    gcn_rebuild_cache &rc = rebuild_cache(d);
    cudaStream_t s = d->stream;
    const gcn_term_copy &m = rc.real_mask;
    const gc_i64 nm = m.nt > 0 ? m.nt : 0;
    *rows = 0;
    *index = 0;
    *n = 0;
    rc.mnsel = 0;
    const bool pre = rc.mpre_room >= 0 && !member && nq > 0 &&
                     (gc_i64)rc.mpre_q.size() == nq &&
                     std::memcmp(rc.mpre_q.data(), q,
                                 (size_t)nq * sizeof(gc_gid)) == 0;
    const gc_i64 pre_room = rc.mpre_room;
    rc.mpre_room = -1;
    if (member) { member->row = 0; member->extra_flag = 0; }
    if (nm == 0 || nq == 0) return GC_OK;
    gc_i64 *pin = rebuild_pinned(d);
    if (!pin) return GC_E_NOMEM;
    /* gid pairs, image pairs and list indices, staged apart and read back as
     * rows into the room earlier selections bound (with margin), so one
     * synchronisation returns count and rows; a selection that outgrew it is
     * gathered again. */
    gc_i64 k = 0, room = pre ? pre_room : mask_room(rc, nm, nq);
    const int flags = member && member->table;
    gc_status st = pre ? GC_OK : mask_select_queue(rc, d, q, nq, member, room);
    if (st != GC_OK) return st;
    for (int pass = 0; ; ++pass) {
        if (pass > 0) {
            st = mask_gather_queue(rc, d, nq, member, room);
            if (st != GC_OK) return st;
        }
        st = dev_phase(s, "mask_select");
        if (st != GC_OK) return st;
        k = pin[GCN_PIN_N - 8];
        if (k < 0 || k > nm || (pass > 0 && k > room)) return GC_E_STATE;
        if (k <= room) break;
        room = k;
    }
    rc.mnsel = k;
    rc.mrate = std::max(rc.mrate, (double)k / (double)nq);
    if (flags) {
        const gc_u8 *f = (const gc_u8 *)rc.mstage +
                         room * (gc_i64)(sizeof(gcn_mask_row) + sizeof(gc_i64));
        member->row = f;
        member->extra_flag = f + 2 * room;
    }
    if (k == 0) return GC_OK;
    std::vector<gcn_mask_row> tmp((size_t)k);
    const gc_gid *hg = (const gc_gid *)rc.mstage;
    const gc_image *hi = (const gc_image *)(hg + 2 * room);
    for (gc_i64 i = 0; i < k; ++i) {
        tmp[(size_t)i].a = hg[2 * i];     tmp[(size_t)i].ia = hi[2 * i];
        tmp[(size_t)i].b = hg[2 * i + 1]; tmp[(size_t)i].ib = hi[2 * i + 1];
    }
    std::memcpy(rc.mstage, tmp.data(), tmp.size() * sizeof(gcn_mask_row));
    *rows = (const gcn_mask_row *)rc.mstage;
    *index = (const gc_i64 *)(hi + 2 * room);
    *n = k;
    return GC_OK;
}

/* Replace the rows the last native_mask_select selected by the n rows
 * given: the unselected keep their order, the given ones append. */
gc_status native_mask_replace(gc_context *ctx, const gcn_mask_row *rows,
                              gc_i64 n)
{
    struct gcn_device *d = ctx ? ctx->native : 0;
    if (!ctx || !d || n < 0 || (n > 0 && !rows)) return GC_E_ARG;
    gcn_rebuild_cache &rc = rebuild_cache(d);
    cudaStream_t s = d->stream;
    gcn_term_copy &t = rc.real_mask, &a = rc.mask_alt;
    const gc_i64 nm = t.nt > 0 ? t.nt : 0;
    const gc_i64 w = nm - rc.mnsel, m = w + n;
    gc_status st = grow((void **)&a.ep, &a.ep_cap, 2 * m * (gc_i64)sizeof(gc_gid));
    if (st == GC_OK) st = grow((void **)&a.epi, &a.epi_cap, 2 * m * (gc_i64)sizeof(gc_image));
    if (st != GC_OK) return st;
    if (rc.mnsel > 0)
        gcn_kern_term_compact<<<(unsigned)grid_for(nm, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
            rc.msel, rc.mpos, nm, 2, 0, t.ep, t.epi, (const gc_i32 *)0,
            (const gc_i32 *)0, (const gc_i32 *)0, a.ep, a.epi, (gc_i32 *)0,
            (gc_i32 *)0, (gc_i32 *)0);
    else if (nm > 0 &&
             (cudaMemcpyAsync(a.ep, t.ep, (size_t)(2 * nm) * sizeof(gc_gid),
                              cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
              cudaMemcpyAsync(a.epi, t.epi, (size_t)(2 * nm) * sizeof(gc_image),
                              cudaMemcpyDeviceToDevice, s) != cudaSuccess))
        return GC_E_DEVICE;
    if (n > 0) {
        std::vector<gc_gid> g((size_t)(2 * n));
        std::vector<gc_image> im((size_t)(2 * n));
        for (gc_i64 i = 0; i < n; ++i) {
            g[(size_t)(2 * i)] = rows[i].a;     im[(size_t)(2 * i)] = rows[i].ia;
            g[(size_t)(2 * i + 1)] = rows[i].b; im[(size_t)(2 * i + 1)] = rows[i].ib;
        }
        if (cudaMemcpyAsync(a.ep + 2 * w, g.data(), g.size() * sizeof(gc_gid),
                            cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(a.epi + 2 * w, im.data(), im.size() * sizeof(gc_image),
                            cudaMemcpyHostToDevice, s) != cudaSuccess)
            return GC_E_DEVICE;
    }
    st = rb_phase(s, "mask_replace");
    if (st != GC_OK) return st;
    std::swap(a, t);
    t.nt = m;
    t.valid = true;
    a.valid = false;
    rc.mnsel = 0;
    rc.mask_dev = true;
    return GC_OK;
}

gc_status native_upload_import(gc_context *ctx)
{
    struct gcn_device *d = ctx->native;
    const gc_i64 n = ctx->num_owned;
    const gc_i64 ng = ctx->num_ghost;
    const gc_i64 p = d->pitch;
    term_copies_invalidate(d);
    native_migration_release(ctx);

    std::vector<gc_f64> tmp((size_t)(3 * p), 0.0);

#define GCN_PUSH3(src, dst)                                                  \
    do {                                                                     \
        for (gc_i64 i = 0; i < n; ++i)                                       \
            for (int c = 0; c < 3; ++c)                                      \
                tmp[(size_t)(c * p + i)] = (src)[3 * i + c];                 \
        if (cudaMemcpy((dst), &tmp[0], (size_t)(3 * p) * sizeof(gc_f64),     \
                       cudaMemcpyHostToDevice) != cudaSuccess)               \
            return GC_E_DEVICE;                                              \
    } while (0)

    GCN_PUSH3(ctx->atom_coord, d->coord);
    GCN_PUSH3(ctx->atom_coord, d->coord_ref);
    GCN_PUSH3(ctx->atom_coord, d->list_ref);
    GCN_PUSH3(ctx->atom_vel,   d->vel_ref);
#undef GCN_PUSH3
    if (native_vf_put(d, d->vel, &tmp[0]) != GC_OK) return GC_E_DEVICE;

#define GCN_PUSH1(vec, dst, type)                                            \
    do {                                                                     \
        if (cudaMemcpy((dst), &(vec)[0], (size_t)n * sizeof(type),           \
                       cudaMemcpyHostToDevice) != cudaSuccess)               \
            return GC_E_DEVICE;                                              \
    } while (0)

    GCN_PUSH1(ctx->atom_charge,   d->charge,   gc_f64);
    GCN_PUSH1(ctx->atom_mass,     d->mass,     gc_f64);
    GCN_PUSH1(ctx->atom_inv_mass, d->inv_mass, gc_f64);
    GCN_PUSH1(ctx->atom_cls,      d->cls,      gc_i32);
    GCN_PUSH1(ctx->atom_gid,      d->gid,      gc_gid);
#undef GCN_PUSH1

    if (ng > 0) {
        std::vector<gc_f64> g3((size_t)(3 * ng));
        for (gc_i64 i = 0; i < ng; ++i)
            for (int c = 0; c < 3; ++c)
                g3[(size_t)(c * ng + i)] = ctx->ghost_coord[(size_t)(3 * i + c)];
        for (int c = 0; c < 3; ++c) {
            if (cudaMemcpy(d->coord + (gc_i64)c * p + n,
                           &g3[(size_t)(c * ng)], (size_t)ng * sizeof(gc_f64),
                           cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;
            if (cudaMemcpy(d->force_coord + (gc_i64)c * p + n,
                           &g3[(size_t)(c * ng)], (size_t)ng * sizeof(gc_f64),
                           cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;
        }
        if (cudaMemcpy(d->charge + n, &ctx->ghost_charge[0],
                       (size_t)ng * sizeof(gc_f64), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d->cls + n, &ctx->ghost_cls[0],
                       (size_t)ng * sizeof(gc_i32), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d->gid + n, &ctx->ghost_gid[0],
                       (size_t)ng * sizeof(gc_gid), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d->image + n, &ctx->ghost_image[0],
                       (size_t)ng * sizeof(gc_image), cudaMemcpyHostToDevice) != cudaSuccess)
            return GC_E_DEVICE;
    }

    std::vector<gc_image> himage((size_t)d->num_resident, 0);
    std::vector<gc_i32> howner((size_t)d->num_resident, ctx->rank);
    std::vector<gc_i32> hcell((size_t)d->num_resident, -1);
    if (ctx->atom_cell.size() != (size_t)n) return GC_E_ARG;
    for (gc_i64 i = 0; i < n; ++i) {
        const gc_i32 hc = ctx->atom_cell[(size_t)i];
        if (hc < 0 || (gc_i64)hc >= ctx->ncell_local) return GC_E_ENDPOINT;
        const gc_i32 c = gcn_box_index(d->layout,
            ctx->cell_gx[(size_t)hc] - 1,
            ctx->cell_gy[(size_t)hc] - 1,
            ctx->cell_gz[(size_t)hc] - 1);
        if (c < 0 || !gcn_box_owned(d->layout, c)) return GC_E_OWNER;
        hcell[(size_t)i] = c;
    }
    for (gc_i64 i = 0; i < ng; ++i) {
        const gc_i32 hc = ctx->ghost_cell[(size_t)i];
        if (hc < 0 || (gc_i64)hc >= ctx->ncell) return GC_E_ENDPOINT;
        const gc_i32 ex = ctx->cell_gx[(size_t)hc] - 1;
        const gc_i32 ey = ctx->cell_gy[(size_t)hc] - 1;
        const gc_i32 ez = ctx->cell_gz[(size_t)hc] - 1;
        const gc_i32 c = gcn_box_index(d->layout, ex, ey, ez);
        if (c < 0) return GC_E_ENDPOINT;
        gc_i32 ix, iy, iz;
        const gc_i32 gx = gcn_wrap_axis(ex, d->layout.ncel[0], &ix);
        const gc_i32 gy = gcn_wrap_axis(ey, d->layout.ncel[1], &iy);
        const gc_i32 gz = gcn_wrap_axis(ez, d->layout.ncel[2], &iz);
        const gc_image expect = ((gc_image)(gc_i16)ix & 0xffffLL)
                              | (((gc_image)(gc_i16)iy & 0xffffLL) << 16)
                              | (((gc_image)(gc_i16)iz & 0xffffLL) << 32);
        if (expect != ctx->ghost_image[(size_t)i]) return GC_E_MISMATCH;
        const gc_i64 s = n + i;
        himage[(size_t)s] = expect;
        howner[(size_t)s] = gcn_rank_of_cell(d->layout, gx, gy, gz);
        hcell[(size_t)s] = c;
        if (howner[(size_t)s] == ctx->rank) return GC_E_OWNER;
    }
    if (cudaMemcpy(d->image, &himage[0], (size_t)d->num_resident * sizeof(gc_image),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d->owner, &howner[0], (size_t)d->num_resident * sizeof(gc_i32),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d->cell_of, &hcell[0], (size_t)d->num_resident * sizeof(gc_i32),
                   cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;

    if (cudaMemcpy(d->group_offset, &ctx->group_offset[0],
                   (size_t)(ctx->num_groups + 1) * sizeof(gc_i64),
                   cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;
    if (cudaMemcpy(d->group_member, &ctx->group_member[0],
                   (size_t)ctx->num_group_members * sizeof(gc_i32),
                   cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;
    if (cudaMemcpy(d->group_kind, &ctx->group_kind[0],
                   (size_t)ctx->num_groups * sizeof(gc_u8),
                   cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;
    if (cudaMemcpy(d->group_gid, &ctx->group_gid[0],
                   (size_t)ctx->num_groups * sizeof(gc_gid),
                   cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;
    if (cudaMemcpy(d->group_cell, &ctx->group_cell[0],
                   (size_t)ctx->num_groups * sizeof(gc_i32),
                   cudaMemcpyHostToDevice) != cudaSuccess) return GC_E_DEVICE;

    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        const gc_status st = publish_term_pool(ctx, d, k);
        if (st) return st;
    }

    for (int c = 0; c < 3; ++c) {
        d->box[c] = ctx->geo.system_size[c];
        d->cell_count[c] = ctx->geo.cell[c];
    }
    d->pairlistdist2 = ctx->geo.pairlistdist * ctx->geo.pairlistdist;
    return GC_OK;
}

gc_status native_context_attach(gc_context *ctx)
{
    if (!ctx) return GC_E_ARG;
    gc_status st = native_dist_prepare_outer(ctx);
    if (st != GC_OK) return st;
    st = native_state_alloc(ctx);
    st = native_dist_vote(ctx, st, "attach_alloc");
    if (st != GC_OK) { native_state_free(ctx); return st; }
    st = native_upload_import(ctx);
    st = native_dist_vote(ctx, st, "attach_import");
    if (st != GC_OK) { native_state_free(ctx); return st; }
    st = native_dist_create(ctx);
    st = native_dist_vote(ctx, st, "attach_dist");
    if (st != GC_OK) { native_state_free(ctx); return st; }
    return GC_OK;
}

gc_status native_context_release(gc_context *ctx)
{
    return native_state_free(ctx);
}

/* ---- the rebuild --------------------------------------------------- */

/* allow_migration == 0 is the classification-only caller: it runs the
 * classification and pack, then refuses GC_E_OWNER leaving the published
 * state untouched. The product step path passes 1. */
/* The last build's execution records may be carried through this build's
 * permutation: one rank, no ghost band, no migration, every kind's (and the
 * real mask's) canonical copy still the resident one of the same host arrays,
 * with the same counts as when the records were written. */
static int exec_records_carry(const gc_context *ctx, const struct gcn_device *d,
                              int migrated)
{
    if (ctx->nproc != 1 || migrated || d->num_resident != d->num_owned ||
        d->cell_count[0] > 1023 || d->cell_count[1] > 1023 ||
        d->cell_count[2] > 1023)
        return 0;
    auto it = g_rebuild_cache.find(d);
    if (it == g_rebuild_cache.end()) return 0;
    const gcn_rebuild_cache &rc = it->second;
    if (!rc.exec_ok || rc.exec_nres != d->num_resident) return 0;
    for (int k = 0; k < GC_TERM_NKIND; ++k) {
        const gc_i64 nt = d->term_count[k];
        if (nt != rc.exec_nt[k]) return 0;
        if (nt <= 0) continue;
        const gcn::TermPool &tp = ctx->term[k];
        const gcn_term_copy &tc = rc.term[k];
        const gc_i32 ncodes = gc_term_pbc_codes[k];
        if (!tc.valid || tc.nt != nt || tp.endpoint.empty() ||
            tp.endpoint_image.empty() || tp.param_id.empty() ||
            (ncodes > 0 && tp.pbc.empty()) ||
            tc.h_ep != (const void *)&tp.endpoint[0] ||
            tc.h_epi != (const void *)&tp.endpoint_image[0] ||
            tc.h_pid != (const void *)&tp.param_id[0] ||
            tc.h_pbc != (ncodes > 0 ? (const void *)&tp.pbc[0] : 0))
            return 0;
    }
    const gc_i64 nm = (gc_i64)(ctx->real_mask_gid.size() / 2);
    if (nm != rc.exec_nm) return 0;
    if (nm > 0) {
        const gcn_term_copy &mc = rc.real_mask;
        if (!mc.valid || mc.nt != nm ||
            mc.h_ep != (const void *)&ctx->real_mask_gid[0] ||
            mc.h_epi != (const void *)&ctx->real_mask_image[0] ||
            rc.exec_rm == 0 || rc.sc[GCN_SC_RM_EX] != rc.exec_rm ||
            rc.cap[GCN_SC_RM_EX] < nm * (gc_i64)sizeof(struct gcn_term_exec))
            return 0;
    }
    return 1;
}

/* Sort n (hi, lo, value) keys from k* into o* by the bucket sort
 * (gcn_kern_gsort_rank's note): nb cells, the cell at hi >> sh. */
static gc_status bucket_sort(struct gcn_device *d, cudaStream_t s, gc_i64 n,
                             gc_i64 nb, int sh, const gc_u64 *kh,
                             const gc_u64 *kl, const gc_i32 *kv, gc_u64 *oh,
                             gc_u64 *ol, gc_i32 *ov)
{
    const gc_i64 nbk = nb + 1;
    const gc_i64 bytes = 2 * nbk * (gc_i64)sizeof(gc_i32) +
                         nbk * (gc_i64)sizeof(gc_i64) + n * (gc_i64)sizeof(gc_i32);
    char *gb = 0;
    gc_status st = scratch(d, GCN_SC_GSORT, bytes, s, 0, (void **)&gb);
    if (st != GC_OK) return st;
    gc_i64 *boff = (gc_i64 *)gb;
    gc_i32 *cnt = (gc_i32 *)(boff + nbk);
    gc_i32 *cursor = cnt + nbk;
    gc_i32 *tmp = cursor + nbk;
    if (cudaMemsetAsync(cnt, 0, 2 * nbk * sizeof(gc_i32), s) != cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_gsort_count<<<(unsigned)grid_for(n, GCN_BLOCK), GCN_BLOCK,
                           0, s>>>(kh, n, nb, sh, cnt);
    st = mscan(d, (const gc_i32 *)cnt, boff, nbk, (gc_i64 *)0, s);
    if (st != GC_OK) return st;
    gcn_kern_gsort_scatter<<<(unsigned)grid_for(n, GCN_BLOCK), GCN_BLOCK,
                             0, s>>>(kh, n, nb, sh, boff, cursor, tmp);
    gcn_kern_gsort_rank<<<(unsigned)grid_for(32 * nbk, GCN_BLOCK), GCN_BLOCK,
                          0, s>>>(kh, kl, kv, tmp, boff, nbk, n, oh, ol, ov);
    return GC_OK;
}

/* The cell stage's verdict, from the pinned words of its readback: the
 * halo refresh's register and map failures (pin[13]), the deferred force
 * frame's (pin[12]), the resident publication's (pin[6]) and the cell
 * runs' total (pin[7]). */
/* A migrating build's second classification, from pinned words 44..47:
 * the key's failures and the owner check's (moving, moving atoms,
 * invalid), all zero once every group is at its owner. */
static gc_status reclassify_verdict(gc_context *ctx, struct gcn_device *d,
                                    const gc_i64 *pin)
{
    const gc_i64 malformed = pin[44], na = d->num_owned;
    const gc_i64 *after = pin + 45;
    if (!malformed && d->num_members == na &&
        after[0] == 0 && after[1] == 0 && after[2] == 0)
        return GC_OK;
    std::fprintf(stderr,
        "GPU_Core_Error> phase=rebuild_migration_classify rank=%d "
        "malformed=%lld members=%lld owned=%lld moving=%lld "
        "moving_atoms=%lld invalid=%lld\n",
        (int)ctx->rank, (long long)malformed,
        (long long)d->num_members, (long long)na,
        (long long)after[0], (long long)after[1],
        (long long)after[2]);
    return GC_E_STATE;
}

static gc_status candidate_verdict(gc_context *ctx, struct gcn_device *d,
                                   const gc_i64 *pin, int reclassify)
{
    if (reclassify) {
        const gc_status rs = reclassify_verdict(ctx, d, pin);
        if (rs != GC_OK) return rs;
    }
    if (pin[13]) {
        std::fprintf(stderr, "GPU_Core_Error> phase=rebuild_halo_refresh "
                     "rank=%d register_or_map_failures=%lld\n", ctx->rank,
                     (long long)pin[13]);
        return GC_E_MISMATCH;
    }
    if (pin[12]) {
        std::fprintf(stderr,
            "GPU_Core_Error> phase=force_coordinates invalid_groups_or_atoms=%lld\n",
            (long long)pin[12]);
        return GC_E_UNSUPPORTED;
    }
    d->force_move_valid = 1;
    if (pin[6] != 0 || pin[7] != d->num_resident) return GC_E_MISMATCH;
    return GC_OK;
}

/* The compact records of this build's two-endpoint kinds, from the full
 * ones every rebuild path has just left in d->term */
static gc_status native_term_pairs(struct gcn_device *d, cudaStream_t s)
{
    static const int kinds[3] = { GC_TERM_BOND, GC_TERM_NB14, GC_TERM_EXCL };
    d->excl_keep_valid = 0;
    for (int j = 0; j < 3; ++j) {
        const int k = kinds[j];
        const gc_i64 n = d->term_count[k];
        d->term_pair_n[k] = n;
        if (n <= 0) continue;
        const gc_i64 bytes = n * (gc_i64)sizeof(struct gcn_term_pair);
        if (bytes > d->term_pair_cap[k]) {
            dev_release((void **)&d->term_pair[k]);
            d->term_pair_cap[k] = 0;
            const gc_status st = dev_calloc((void **)&d->term_pair[k], bytes + bytes / 4);
            if (st != GC_OK) return st;
            d->term_pair_cap[k] = bytes + bytes / 4;
        }
        gcn_kern_term_pair<<<(unsigned)grid_for(n, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
            d->term[k], n, d->term_pair[k]);
    }
    return dev_launched("term_pairs");
}

static gc_status native_rebuild_impl(gc_context *ctx, gc_i32 early,
                                     gc_i32 allow_migration)
{
    (void)early;  /* the public transaction publishes its class on success */
    struct gcn_device *d = ctx->native;
    cudaStream_t s = d->stream;
    gc_status st;
    d->view_ready = 0;          /* the view is formed again after the build */

    gc_i64 ng = d->num_groups;
    gc_i64 na = d->num_owned;

    /* Device words, zeroed per build: [0] the force frame's failures,
     * [1] the longest cell run, [2] the carry's permutation failures,
     * [3] the carry audit's differing records, [4..6] the selection
     * count's pair queue, oversize-pair count and block queue. */
    gc_i64 *flags = 0;
    st = scratch(d, GCN_SC_FLAGS, 16 * (gc_i64)sizeof(gc_i64), s, 1,
                 (void **)&flags);
    if (st != GC_OK) return st;

    if (cudaMemsetAsync(d->fail_counter,0,sizeof(gc_i64),s)!=cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_group_key<<<(unsigned)grid_for(ng, GCN_BLOCK), GCN_BLOCK, 0, s>>>(
        d->coord, d->group_offset, d->group_member, d->group_kind,
        d->group_gid, d->group_cell2, d->group_dest_rank,
        d->group_dest_coord, d->sort_key_hi, d->sort_key_lo,
        d->sort_val, ng, d->num_members, na, d->pitch,d->fail_counter,
        ctx->geo.origin[0], ctx->geo.origin[1], ctx->geo.origin[2],
        d->box[0], d->box[1], d->box[2],
        ctx->geo.cell_size[0], ctx->geo.cell_size[1], ctx->geo.cell_size[2],
        d->cell_count[0], d->cell_count[1], d->cell_count[2], d->layout);

    gc_i64 *pin = rebuild_pinned(d);
    if (!pin) return GC_E_NOMEM;
    if (cudaMemsetAsync(d->migration_counts, 0,
                        3 * sizeof(gc_i64), s) != cudaSuccess)
        return GC_E_DEVICE;
    gcn_kern_validate_group_owner<<<(unsigned)grid_for(ng, GCN_BLOCK), GCN_BLOCK,
                                    0, s>>>(d->group_cell2,
                                           d->group_dest_rank,
                                           d->group_dest_coord,
                                           d->group_offset, ng, d->layout,
                                           d->migration_counts);
    static_assert(64 + GCN_VERDICT_W + 2 <= GCN_PIN_N, "pinned words");
    /* The need vote (any rank with a moving group) is summed on the device on
     * one node (gcx_wsum) and comes back with this synchronisation. */
    gcx_wsum *const need_sum = ctx->nproc > 1 ? native_dist_wsum(ctx) : 0;
    gc_i64 *vpin = pin + 64, *fpin = pin + 64 + GCN_VERDICT_W;
    fpin[0] = 0; fpin[1] = 0;
    if (cudaMemcpyAsync(pin, d->fail_counter, sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaMemcpyAsync(pin + 1, d->migration_counts, 3 * sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        (need_sum &&
         (cudaMemcpyAsync(flags + 9, d->migration_counts, sizeof(gc_i64),
                          cudaMemcpyDeviceToDevice, s) != cudaSuccess ||
          gcx_wsum_launch(need_sum, (gc_u64 *)(flags + 9), (void *)s) != GC_OK ||
          cudaMemcpyAsync(pin + 15, flags + 9, sizeof(gc_i64),
                          cudaMemcpyDeviceToHost, s) != cudaSuccess)) ||
        cudaMemcpyAsync(vpin, d->verdict, GCN_VERDICT_W * sizeof(gc_i64),
                        cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        (d->con_fail &&
         cudaMemcpyAsync(fpin, d->con_fail, 2 * sizeof(gc_i64),
                         cudaMemcpyDeviceToHost, s) != cudaSuccess))
        return GC_E_DEVICE;
    st = dev_phase(s, "group_key");
    if (st == GC_OK) {
        const gc_status vs = native_verdict_check(ctx, vpin, fpin);
        if (vs != GC_OK) return vs;
    }
    gc_i64 malformed = (st == GC_OK) ? pin[0] : 0;
    const gc_i64 candidate[3] = { pin[1], pin[2], pin[3] };
    if(d->num_members!=na || (ng==0 && na!=0))++malformed;
    if(st==GC_OK && malformed) {
        std::fprintf(stderr,
            "GPU_Core_Error> phase=rebuild_group_csr malformed=%lld "
            "kernel_bad=%lld groups=%lld members=%lld owned=%lld\n",
            (long long)malformed,
            (long long)(malformed - ((d->num_members != na ||
                                      (ng == 0 && na != 0)) ? 1 : 0)),
            (long long)ng, (long long)d->num_members, (long long)na);
        st=GC_E_ARITY;
    }
    st=native_dist_step_vote(ctx,st,"rebuild_group_csr");
    if (st != GC_OK) return st;
    /* Every rank enters the migration when any rank has a mover. Where the
     * need is not yet known, a rank with a mover queues its pack before the
     * host vote completes: the pack writes only the transaction's inputs,
     * which nothing reads unless the transaction runs. */
    const int product = 1;
    const int early_pack = product && !need_sum && allow_migration &&
                           ctx->nproc > 1 && candidate[0] > 0 &&
                           ctx->epoch >= 0 &&
                           ctx->epoch != std::numeric_limits<gc_i64>::max();
    struct gcn_dist_max_req need_vote;
    gc_i64 need = 0;
    if (need_sum && product) {
        need_vote.active = 0;
        need = pin[15];
    } else {
        native_dist_max_start(ctx, st == GC_OK ? pin[1] : 1, &need_vote);
        need = early_pack ? 1 : native_dist_max_wait(&need_vote);
    }
    int need_bad = 0;
    gc_i64 migrate = need > 0 ? candidate[0] : 0;
    st = native_dist_step_vote(ctx,
                               (need_bad || candidate[2] || candidate[1] > na) ?
                               GC_E_MISMATCH : GC_OK,
                               "rebuild_group_candidate");
    if (st != GC_OK) {
        native_dist_max_wait(&need_vote);
        return st;
    }
    st = cudaMemsetAsync(d->migration_pack_totals,0,
                         3*sizeof(gc_i64),s)==cudaSuccess ? GC_OK : GC_E_DEVICE;
    /* Clear the per-peer counts and segment bases with the totals: only the
     * offsets kernel writes them, so a rank with nothing to send must state
     * zero for this rebuild. */
    if (st == GC_OK &&
        (cudaMemsetAsync(d->migration_peer_counts, 0,
                         3 * (size_t)ctx->nproc * sizeof(gc_i64), s) !=
             cudaSuccess ||
         cudaMemsetAsync(d->migration_peer_group_base, 0,
                         (size_t)ctx->nproc * sizeof(gc_i64), s) !=
             cudaSuccess ||
         cudaMemsetAsync(d->migration_peer_atom_base, 0,
                         (size_t)ctx->nproc * sizeof(gc_i64), s) !=
             cudaSuccess))
        st = GC_E_DEVICE;
    if (st == GC_OK && !migrate) st = rb_phase(s, "group_migration_clear");
    const int pack_to_txn = early_pack || (product && need_sum);
    struct gcn_pack_check pack = {};
    if (migrate) {
        if (st == GC_OK &&
            (ctx->epoch < 0 ||
             ctx->epoch == std::numeric_limits<gc_i64>::max()))
            st = GC_E_EPOCH;
        if (st == GC_OK &&
            cudaMemsetAsync(d->fail_counter,0,sizeof(gc_i64),s)!=cudaSuccess)
            st = GC_E_DEVICE;
        char *mb = 0;
        if (st == GC_OK)
            st = scratch(d, GCN_SC_MIGLIST,
                         ng * (gc_i64)(2 * sizeof(gc_i32) + sizeof(gc_i64)) +
                         (gc_i64)sizeof(gc_i64), s, 0, (void **)&mb);
        if (st == GC_OK) {
            gc_i64 *at = (gc_i64 *)mb, *nmover = at + ng;
            gc_i32 *mover = (gc_i32 *)(nmover + 1);
            gcn_kern_migration_count<<<(unsigned)grid_for(ng,GCN_BLOCK),
                                       GCN_BLOCK,0,s>>>(
                d->group_dest_rank,d->group_offset,ng,na,ctx->rank,
                d->layout.nproc,
                d->migration_group_offset,d->migration_atom_offset,
                d->migration_pack_totals,d->migration_peer_counts,mover);
            st = mscan(d, (const gc_i32 *)mover, at, ng, nmover, s);
        }
        if (st == GC_OK) {
            gc_i64 *at = (gc_i64 *)mb, *nmover = at + ng;
            gc_i32 *mover = (gc_i32 *)(nmover + 1), *list = mover + ng;
            gcn_kern_migration_list<<<(unsigned)grid_for(ng,GCN_BLOCK),
                                      GCN_BLOCK,0,s>>>(mover, at, ng, list);
            gcn_kern_migration_layout<<<1,1,0,s>>>(
                d->layout.nproc,
                d->migration_pack_totals,d->migration_peer_counts,
                d->migration_peer_group_base,d->migration_peer_atom_base);
            gcn_kern_migration_places<<<(unsigned)d->layout.nproc,32,0,s>>>(
                d->group_dest_rank,d->group_offset,list,nmover,
                d->layout.nproc,d->migration_pack_totals,
                d->migration_peer_group_base,d->migration_peer_atom_base,
                d->migration_group_offset,d->migration_atom_offset);
            vf_each(d->vf32, [&](auto z) {
            using S = decltype(z);
            gcn_kern_migration_pack<<<(unsigned)grid_for(ng,GCN_BLOCK),
                                      GCN_BLOCK,0,s>>>(
                d->group_offset,d->group_member,d->group_kind,d->group_gid,
                d->group_dest_rank,d->group_dest_coord,
                d->migration_group_offset,d->migration_atom_offset,
                d->migration_peer_atom_base,
                d->gid,d->cls,d->charge,d->mass,d->inv_mass,
                d->coord,d->coord_ref,d->list_ref,d->vel.as<S>(),
                d->vel_ref,d->vel_half,d->vel_full,d->force.as<S>(),
                ng,na,d->pitch,ctx->epoch+1,ctx->rank,
                d->num_groups,d->num_owned,
                d->migration_group_send,d->migration_atom_send,
                d->fail_counter);
            });
            if (cudaMemcpyAsync(pin + 36,d->migration_pack_totals,
                                3*sizeof(gc_i64),cudaMemcpyDeviceToHost,s)!=cudaSuccess ||
                cudaMemcpyAsync(pin + 39,d->fail_counter,sizeof(gc_i64),
                                cudaMemcpyDeviceToHost,s)!=cudaSuccess)
                st = GC_E_DEVICE;
            pack.packed = pin + 36;
            for (int k = 0; k < 3; ++k) pack.candidate[k] = candidate[k];
            pack.groups = ng;
            pack.owned = na;
            if (st == GC_OK && !pack_to_txn) {
                st = dev_phase(s,"group_migration_pack");
                if (st == GC_OK) st = native_pack_verdict(ctx, &pack);
            }
        }
    }
    if (early_pack)     /* this rank's own movers decided the vote */
        native_dist_max_wait(&need_vote);
    st = native_dist_step_vote(ctx, st, "rebuild_group_pack");
    if (st != GC_OK) return st;

    /* The real transaction is collective: every rank enters it whenever any
     * rank found a moving group, because counts and payload are exchanged on
     * the same edges. A refusal returns with the epoch, terms, groups and
     * published CSR unchanged. */
    gc_i32 migrated = 0;
    if (!allow_migration) {
        st = native_dist_vote(ctx, migrate ? GC_E_OWNER : GC_OK,
                              "rebuild_group_migration");
        if (st != GC_OK) {
            if (migrate)
                std::fprintf(stderr,
                    "GPU_Core_Error> phase=rebuild_migration pending_groups="
                    "%lld pending_atoms=%lld reason=candidate migration not "
                    "published (classification-only caller)\n",
                    (long long)migrate,(long long)candidate[1]);
            return st;
        }
    } else if (ctx->nproc > 1 && need > 0) {
        gc_i64 moved_groups = 0, moved_atoms = 0;
        st = gcn::migration_txn_native_run(ctx, &moved_groups, &moved_atoms,
                                           &migrated,
                                           pack_to_txn && migrate ? &pack : 0);
        if (st != GC_OK) {
            std::fprintf(stderr,
                "GPU_Core_Error> phase=rebuild_migration status=%d "
                "reason=migration transaction refused\n", (int)st);
            return st;
        }
    } else if (migrate) {
        return GC_E_OWNER;
    }
    if (migrated) d->rigid_ready = 0;
    const int kDeferClassify = 1;
    int defer_classify = 0;
    if (migrated) {
        ng = d->num_groups;
        na = d->num_owned;
        if (cudaMemsetAsync(d->fail_counter, 0, sizeof(gc_i64), s) !=
                cudaSuccess ||
            cudaMemsetAsync(d->migration_counts, 0, 3 * sizeof(gc_i64), s) !=
                cudaSuccess)
            return GC_E_DEVICE;
        gcn_kern_group_key<<<(unsigned)grid_for(ng, GCN_BLOCK), GCN_BLOCK,
                              0, s>>>(
            d->coord, d->group_offset, d->group_member, d->group_kind,
            d->group_gid, d->group_cell2, d->group_dest_rank,
            d->group_dest_coord, d->sort_key_hi, d->sort_key_lo,
            d->sort_val, ng, d->num_members, na, d->pitch, d->fail_counter,
            ctx->geo.origin[0], ctx->geo.origin[1], ctx->geo.origin[2],
            d->box[0], d->box[1], d->box[2],
            ctx->geo.cell_size[0], ctx->geo.cell_size[1],
            ctx->geo.cell_size[2],
            d->cell_count[0], d->cell_count[1], d->cell_count[2], d->layout);
        defer_classify = kDeferClassify;
        gc_i64 *kpin = rebuild_pinned(d);
        st = kpin ? GC_OK : GC_E_NOMEM;
        if (st == GC_OK &&
            cudaMemsetAsync(d->migration_counts, 0, 3 * sizeof(gc_i64), s) !=
                cudaSuccess)
            st = GC_E_DEVICE;
        if (st == GC_OK)
            gcn_kern_validate_group_owner<<<(unsigned)grid_for(ng, GCN_BLOCK),
                                            GCN_BLOCK, 0, s>>>(
                d->group_cell2, d->group_dest_rank, d->group_dest_coord,
                d->group_offset, ng, d->layout, d->migration_counts);
        if (st == GC_OK &&
            (cudaMemcpyAsync(kpin + 44, d->fail_counter, sizeof(gc_i64),
                             cudaMemcpyDeviceToHost, s) != cudaSuccess ||
             cudaMemcpyAsync(kpin + 45, d->migration_counts, 3 * sizeof(gc_i64),
                             cudaMemcpyDeviceToHost, s) != cudaSuccess))
            st = GC_E_DEVICE;
        if (st == GC_OK && !defer_classify) {
            st = dev_phase(s, "group_owner_after_migration");
            if (st == GC_OK) st = reclassify_verdict(ctx, d, kpin);
        }
        st = native_dist_step_vote(ctx, st, "rebuild_migration_classify");
        if (st != GC_OK) return st;
    }

    const int carry = exec_records_carry(ctx, d, migrated);
    rebuild_cache(d).exec_ok = false;

    /* Admission passed: from here the current arrays are mutated, so close
     * the old list and captured segment first. */
    d->list_valid = 0;
    if (ctx->nproc > 1) ++d->slot_generation;
    { gc_i32 *old = d->group_cell;
      d->group_cell = d->group_cell2;
      d->group_cell2 = old; }

    gc_u64 *kh = d->sort_key_hi, *kl = d->sort_key_lo;
    gc_u64 *oh = d->sort_key_hi2, *ol = d->sort_key_lo2;
    gc_i32 *kv = d->sort_val, *ov = d->sort_val2;
    if (ng > 1) {
        st = bucket_sort(d, s, ng, d->ncell, 8, kh, kl, kv, oh, ol, ov);
        if (st != GC_OK) return st;
        gc_u64 *t1 = kh; kh = oh; oh = t1;
        gc_u64 *t2 = kl; kl = ol; ol = t2;
        gc_i32 *t3 = kv; kv = ov; ov = t3;
    }

    st = rb_phase(s, "group_sort");
    if (st != GC_OK) return st;

    /* 2. member counts, new offsets, permutation and cell of every slot; the
     * old and new group arrays are distinct buffers that swap. */
    gcn_kern_group_counts<<<(unsigned)grid_for(ng, GCN_BLOCK), GCN_BLOCK,
                            0, s>>>(kv, d->group_offset,
                                    (gc_i64 *)d->scratch3, ng);
    st = mscan(d, (const gc_i64 *)d->scratch3, d->group_offset2, ng,
               (gc_i64 *)0, s);
    if (st != GC_OK) return st;

    gcn_kern_group_scatter<<<(unsigned)grid_for(ng, GCN_BLOCK), GCN_BLOCK,
                             0, s>>>(
        kv, d->group_offset, d->group_member, d->group_kind, d->group_gid,
        d->group_cell, d->group_offset2, d->group_member2, d->group_kind2,
        d->group_gid2, d->group_cell2, d->perm, d->cell_of, ng);

    /* the scan leaves the last offset implicit; at one rank the owned count
     * does not change, so the sentinel is the atom count (a one-thread store:
     * a pageable copy would synchronise) */
    gcn_kern_set_i64<<<1, 1, 0, s>>>(d->group_offset2 + ng, na);

    st = rb_phase(s, "group_scatter");
    if (st != GC_OK) return st;

    { gc_i64 *t0 = d->group_offset; d->group_offset = d->group_offset2;
      d->group_offset2 = t0;
      gc_i32 *t1 = d->group_member; d->group_member = d->group_member2;
      d->group_member2 = t1;
      gc_u8  *t2 = d->group_kind;   d->group_kind   = d->group_kind2;
      d->group_kind2 = t2;
      gc_gid *t3 = d->group_gid;    d->group_gid    = d->group_gid2;
      d->group_gid2 = t3;
      gc_i32 *t4 = d->group_cell;   d->group_cell   = d->group_cell2;
      d->group_cell2 = t4; }

    {
        const struct gcn_perm_set a = {
            { d->coord, d->coord_ref, d->vel.p, d->vel_ref, d->vel_half, d->vel_full },
            d->charge, d->mass, d->inv_mass, d->cls, d->gid };
        const struct gcn_perm_set b = {
            { d->coord2, d->coord_ref2, d->vel2.p, d->vel_ref2, d->vel_half2, d->vel_full2 },
            d->charge2, d->mass2, d->inv_mass2, d->cls2, d->gid2 };
        vf_each(d->vf32, [&](auto z) {
            gcn_kern_permute_all<decltype(z)>
                <<<(unsigned)grid_for(d->num_resident, GCN_BLOCK),
                   GCN_BLOCK, 0, s>>>(a, b, d->perm, na, d->num_resident,
                                      d->pitch);
        });
        std::swap(d->coord, d->coord2);       std::swap(d->coord_ref, d->coord_ref2);
        std::swap(d->vel, d->vel2);           std::swap(d->vel_ref, d->vel_ref2);
        std::swap(d->vel_half, d->vel_half2); std::swap(d->vel_full, d->vel_full2);
        std::swap(d->charge, d->charge2);     std::swap(d->mass, d->mass2);
        std::swap(d->inv_mass, d->inv_mass2);
        std::swap(d->cls, d->cls2);           std::swap(d->gid, d->gid2);
    }

    st = rb_phase(s, "permute");
    if (st != GC_OK) return st;

    gc_i32 *inv = 0;
    if (carry) {
        st = scratch(d, GCN_SC_INV, na * (gc_i64)sizeof(gc_i32), s, 0,
                     (void **)&inv);
        if (st != GC_OK) return st;
        gcn_kern_invert_perm<<<(unsigned)grid_for(na, GCN_BLOCK), GCN_BLOCK,
                               0, s>>>(d->perm, na, inv, flags + 2);
    }

    /* The force frame of the owned slots, in the new slot order: the permute
     * moves the raw state but not force_coord, and the halo refresh packs
     * force_coord for the peer's pair-list prune, so the frame is captured
     * first. Its failure count is read with the cell stage's synchronisation. */
    st = force_coords_launch(ctx, 1, flags + 0, 1);
    if (st != GC_OK) return st;

    /* A migration changes the ghost band's identity: replace the band from
     * the peer's pack and let refresh_maps prove the table before the pair
     * list reads it. Without migration the band's force frame is still stale,
     * so every split build refreshes. */
    if (ctx->nproc > 1) {
        st = gcn::native_dist_halo_refresh(ctx);
        if (st != GC_OK) {
            std::fprintf(stderr,
                "GPU_Core_Error> phase=rebuild_halo_refresh status=%d\n",
                (int)st);
            return st;
        }
    }

    const gc_i32 defer_maps = 1;
    st = native_dist_step_vote(ctx, native_dist_refresh_maps(ctx, defer_maps),
                               "rebuild_maps");
    if (st != GC_OK) return st;

    gcn_kern_copy3<<<(unsigned)grid_for(3 * d->pitch, GCN_BLOCK), GCN_BLOCK,
                     0, s>>>(d->list_ref, d->coord, 3 * d->pitch);

    gc_i32 max_run = 0;   /* the longest cell run: sizes the selection bits */
    gcn_kern_cell_runs<<<(unsigned)grid_for(d->ncell, GCN_BLOCK), GCN_BLOCK,
                         0, s>>>(d->cell_of, d->group_cell, d->cell,
                                 d->num_resident, ng, d->ncell, d->layout);
    {
        /* Resident slots are sorted by (local-box-cell, raw-slot), so every
         * cell run is compact. */
        const gc_i64 nr = d->num_resident;
        gcn_kern_resident_key<<<(unsigned)grid_for(nr, GCN_BLOCK), GCN_BLOCK,
                                0, s>>>(d->cell_of, d->sort_key_hi,
                                       d->sort_key_lo, d->sort_val,
                                       nr, d->ncell);
        gc_u64 *rkh=d->sort_key_hi, *rkl=d->sort_key_lo;
        gc_i32 *rkv=d->sort_val;
        gc_u64 *roh=d->sort_key_hi2, *rol=d->sort_key_lo2;
        gc_i32 *rov=d->sort_val2;
        /* With no ghost band the residents are the owned slots, already in
         * the group sort's order (cell, kind, gid), so the (cell, slot) sort
         * is skipped and the order is checked instead (a violation counts
         * with the publish kernel's failures). */
        const int presorted = (nr == d->num_owned);
        if (!presorted) {
            st = bucket_sort(d, s, nr, d->ncell, 0, rkh, rkl, rkv, roh, rol, rov);
            if (st != GC_OK) return st;
            rkh = roh; rkl = rol; rkv = rov;
        }
        if (cudaMemsetAsync(d->fail_counter, 0, sizeof(gc_i64), s) != cudaSuccess)
            return GC_E_DEVICE;
        if (presorted)
            gcn_kern_key_order_check<<<(unsigned)grid_for(nr, GCN_BLOCK),
                                       GCN_BLOCK, 0, s>>>(rkh, nr,
                                                          d->fail_counter);
        gcn_kern_resident_publish<<<(unsigned)grid_for(nr, GCN_BLOCK), GCN_BLOCK,
                                    0, s>>>(rkh, rkv, d->resident_slot,
                                           d->perm, nr, d->ncell,
                                           d->fail_counter);

        gc_i32 *endpos = 0;
        st = scratch(d, GCN_SC_ENDPOS, d->ncell * (gc_i64)sizeof(gc_i32), s, 1, (void **)&endpos);
        if (st != GC_OK) return st;
        gcn_kern_cell_bounds<<<(unsigned)grid_for(nr, GCN_BLOCK), GCN_BLOCK,
                               0, s>>>(d->perm, nr, d->cell, 0, endpos);
        gcn_kern_cell_counts<<<(unsigned)grid_for(d->ncell, GCN_BLOCK),
                               GCN_BLOCK, 0, s>>>(d->cell, endpos,
                                                  d->ncell, 0);
        cudaMemsetAsync(endpos, 0, (size_t)d->ncell * sizeof(gc_i32), s);
        gcn_kern_cell_bounds<<<(unsigned)grid_for(ng, GCN_BLOCK), GCN_BLOCK,
                               0, s>>>(d->group_cell, ng, d->cell, 1, endpos);
        gcn_kern_cell_counts<<<(unsigned)grid_for(d->ncell, GCN_BLOCK),
                               GCN_BLOCK, 0, s>>>(d->cell, endpos,
                                                  d->ncell, 1);
        gc_i32 *catoms = 0;
        st = scratch(d, GCN_SC_CATOMS, d->ncell * (gc_i64)sizeof(gc_i32), s, 0,
                     (void **)&catoms);
        if (st != GC_OK) { scratch_drop((void **)&endpos); return st; }
        gcn_kern_cell_atoms<<<(unsigned)grid_for(d->ncell, GCN_BLOCK), GCN_BLOCK,
                              0, s>>>(d->cell, d->ncell, catoms, flags + 1);
        st = mscan(d, (const gc_i32 *)catoms, d->resident_offset, d->ncell,
                   d->resident_offset + d->ncell, s);
        scratch_drop((void **)&catoms);
        scratch_drop((void **)&endpos);
        if (st != GC_OK) return st;
    }

    st = rb_phase(s, "cell_runs");
    if (st != GC_OK) return st;

    struct gcn_stencil sten;
    for (int k = 0; k < 3; ++k) {
        double need = ctx->geo.pairlistdist / ctx->geo.cell_size[k];
        int nh = (int)ceil(need - 1.0e-12);
        if (nh < 2) nh = 2;
        if (nh > GCN_MAX_STENCIL || 2 * nh + 1 > d->cell_count[k]) {
            std::fprintf(stderr, "GPU_Core_Error> phase=rebuild_stencil axis=%d "
                         "reach=%d cells=%d max=%d\n", k, nh,
                         (int)d->cell_count[k], GCN_MAX_STENCIL);
            return GC_E_UNSUPPORTED;
        }
        sten.nh[k]   = nh;
        sten.span[k] = 2 * nh + 1;
        sten.ncel[k] = d->cell_count[k];
        sten.box[k]  = d->box[k];
    }
    sten.layout = d->layout;
    sten.size  = sten.span[0] * sten.span[1] * sten.span[2];
    sten.list2 = d->pairlistdist2;

    const gc_i64 nslot = d->ncell * sten.size;
    gc_i32 *keep = 0, *pair_at = 0, *cnt_i = 0, *cnt_j = 0;
    gc_i64 *bits = 0;
    gc_i64 *slot = 0, *off_i = 0, *off_j = 0, *off_m = 0, *totals = 0;
    gc_i64 *report = 0;

    st = scratch(d, GCN_SC_KEEP, nslot * (gc_i64)sizeof(gc_i32), s, 1, (void **)&keep);
    if (st) return st;
    st = scratch(d, GCN_SC_PAIR_AT, nslot * (gc_i64)sizeof(gc_i32), s, 1, (void **)&pair_at);
    if (st) { scratch_drop((void **)&keep); return st; }
    st = scratch(d, GCN_SC_SLOT, nslot * (gc_i64)sizeof(gc_i64), s, 1, (void **)&slot);
    if (st) { scratch_drop((void **)&keep); scratch_drop((void **)&pair_at);
              return st; }
    /* totals[0] is the scan total and every scan writes it, so no counter
     * that must survive a scan may live there. Each writer owns its slots:
     * 0..3 the run-length check, 4..7 the mask-index check, 8..12 the list
     * statistics. */
    st = scratch(d, GCN_SC_TOTALS, 4 * (gc_i64)sizeof(gc_i64), s, 1, (void **)&totals);
    if (st) return st;
    st = scratch(d, GCN_SC_REPORT, 16 * (gc_i64)sizeof(gc_i64), s, 1, (void **)&report);
    if (st) return st;

    gcn_kern_pair_flag<<<(unsigned)grid_for(nslot, GCN_BLOCK), GCN_BLOCK,
                         0, s>>>(d->cell, keep, d->ncell, sten);
    st = mscan(d, (const gc_i32 *)keep, slot, nslot, totals, s);
    if (st != GC_OK) return st;
    /* The list's sizes (pair count, longest cell run, the two selected
     * totals, the mask bits) come back with one synchronisation after the
     * selection count. The count runs into the pair capacity and selection
     * words of the last build; the first build, and a build that outgrows
     * them, reads its pair count and longest run first, grows, and counts
     * again. */
    gcn_rebuild_cache &lc = rebuild_cache(d);
    int presized = d->pair_capacity > 0 && lc.sel_words > 0;
    /* The MIXED cluster-only path reads no selection runs or mask (its pairs
     * carry only the non-empty flag, native_nbc_flags); the DOUBLE and
     * cutoff-cubic real-space kernels keep the full list. */
    const int runs = !native_nbc_active(d);
    gc_i64 npair = 0, sel_words = 0;
    gc_i32 *big = 0;
    gc_u32 *hit = 0;
    gc_u8 *place = 0;
    float sel_lo = 0.f, sel_hi = 0.f;
    double sel_cmax = 0.0;
    gcn_sel_bounds(d->pairlistdist2, d->box, &sel_lo, &sel_hi, &sel_cmax);
    /* After a migration the rigid build's counts ride this readback too. */
    const int rigid_early = d->num_groups > 0 && native_rigid_recount(d);
    st = native_rigid_count_early(ctx);
    if (st != GC_OK) return st;
    for (int pass = 0; ; ++pass) {
        if (cudaMemcpyAsync(pin + 4, totals, sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(pin + 5, flags + 1, sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(pin + 6, d->fail_counter, sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(pin + 7, d->resident_offset + d->ncell, sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(pin + 12, flags + 0, sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess)
            return GC_E_DEVICE;
        pin[13] = 0;
        if (ctx->nproc > 1 &&
            cudaMemcpyAsync(pin + 13, native_dist_verdict(ctx), sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess)
            return GC_E_DEVICE;
        if (!presized) {
            st = dev_phase(s, "pair_candidates");
            if (st != GC_OK) return st;
            st = candidate_verdict(ctx, d, pin, defer_classify);
            if (st != GC_OK) return st;
            npair = pin[4];
            max_run = (gc_i32)pin[5];
            if (npair > d->pair_capacity) {
                dev_release((void **)&d->pair);
                d->pair_capacity = npair + npair / 8 + 16;
                st = dev_calloc((void **)&d->pair,
                                d->pair_capacity * (gc_i64)sizeof(struct gcn_pair));
                if (st) return st;
                dev_release((void **)&d->pair_order);
                st = dev_calloc((void **)&d->pair_order,
                                d->pair_capacity * (gc_i64)sizeof(gc_i32));
                if (st) return st;
            }
            const gc_i64 w = (max_run + 31) / 32 > 0 ? (max_run + 31) / 32 : 1;
            if (w > lc.sel_words) lc.sel_words = w;
        }
        const gc_i64 pcap = presized ? d->pair_capacity : npair;
        const gc_i64 *np_dev = presized ? totals : 0;
        sel_words = lc.sel_words;
        if (pass > 0 &&
            (cudaMemsetAsync(flags + 4, 0, 3 * sizeof(gc_i64), s) != cudaSuccess ||
             cudaMemsetAsync(report, 0, 16 * sizeof(gc_i64), s) != cudaSuccess))
            return GC_E_DEVICE;

        gcn_kern_pair_fill<<<(unsigned)grid_for(nslot, GCN_BLOCK), GCN_BLOCK,
                             0, s>>>(d->cell, keep, slot, pair_at, d->pair,
                                     d->pair_capacity, d->ncell, sten);

        st = rb_phase(s, "pair_fill");
        if (st != GC_OK) return st;

        if (runs) {
        st = scratch(d, GCN_SC_CNT_I, pcap * (gc_i64)sizeof(gc_i32), s, 1, (void **)&cnt_i);
        if (st) return st;
        st = scratch(d, GCN_SC_CNT_J, pcap * (gc_i64)sizeof(gc_i32), s, 1, (void **)&cnt_j);
        if (st) return st;
        st = scratch(d, GCN_SC_OFF_I, pcap * (gc_i64)sizeof(gc_i64), s, 1, (void **)&off_i);
        if (st) return st;
        st = scratch(d, GCN_SC_OFF_J, pcap * (gc_i64)sizeof(gc_i64), s, 1, (void **)&off_j);
        if (st) return st;

        st = scratch(d, GCN_SC_HIT, 2 * pcap * sel_words * (gc_i64)sizeof(gc_u32), s, 0, (void **)&hit);
        if (st) return st;
        st = scratch(d, GCN_SC_PLACE, 2 * pcap * 32 * sel_words, s, 0, (void **)&place);
        if (st) return st;
        }
        if (!runs) {
            /* the clusters and their pair flags (gpu_nbcluster.cu) */
            st = native_nbc_flags(ctx, pcap, np_dev, sel_lo, sel_hi, sel_cmax);
            if (st != GC_OK) return st;
        } else {
            /* hi plus eight float ulps: the box sum may be contracted
               differently from the per-partner sum */
            const float sel_hibox = sel_hi * (1.0f + 8.0f * FLT_EPSILON);
            struct gcn_cbox *cbox = 0;
            st = scratch(d, GCN_SC_CBOX, d->ncell * (gc_i64)sizeof(struct gcn_cbox),
                         s, 0, (void **)&cbox);
            if (st != GC_OK) return st;
            float4 *av = 0;
            st = scratch(d, GCN_SC_AVIEW, d->num_resident * (gc_i64)sizeof(float4),
                         s, 0, (void **)&av);
            if (st != GC_OK) return st;
            gcn_kern_atom_view<<<(unsigned)grid_for(d->num_resident, GCN_BLOCK),
                                 GCN_BLOCK, 0, s>>>(d->force_coord, d->resident_slot,
                                                    d->num_resident, d->pitch,
                                                    0.5 * sel_cmax, av);
            gcn_kern_cell_box<<<(unsigned)grid_for(32 * d->ncell, GCN_BLOCK),
                                GCN_BLOCK, 0, s>>>(av, d->cell, d->ncell, cbox);
            st = scratch(d, GCN_SC_BIG, pcap * (gc_i64)sizeof(gc_i32), s, 0,
                         (void **)&big);
            if (st != GC_OK) return st;
            /* a warp's staging is 5 KB: ask for the largest shared carveout */
            static const cudaError_t carve = cudaFuncSetAttribute(
                gcn_kern_sel_count2, cudaFuncAttributePreferredSharedMemoryCarveout,
                (int)cudaSharedmemCarveoutMaxShared);
            (void)carve;
            gcn_kern_sel_count2<<<pair_grid(pcap), GCN_BLOCK, 0, s>>>(
                d->force_coord, av, d->cell, d->resident_slot, d->pair,
                cnt_i, cnt_j, hit, sel_words, place, report + 15, pcap, np_dev,
                d->pitch, d->pairlistdist2, sel_lo, sel_hi, sel_hibox, cbox,
                (unsigned long long *)(flags + 4), big);
            gcn_kern_sel_count_big<<<pair_grid(pcap), GCN_BLOCK, 0, s>>>(
                d->force_coord, d->cell, d->resident_slot, d->pair, cnt_i, cnt_j,
                hit, sel_words, d->pitch, d->pairlistdist2, sel_lo, sel_hi,
                sel_cmax, (unsigned long long *)(flags + 4), big);
            scratch_drop((void **)&big);
            scratch_drop((void **)&cbox);
            scratch_drop((void **)&av);
        }
        /* Every size the fill and the mask need (selected totals, mask bit
         * total, list statistics) comes back together with one
         * synchronisation; without runs they are zero. */
        if (!runs &&
            cudaMemsetAsync(totals + 1, 0, 3 * sizeof(gc_i64), s) != cudaSuccess)
            return GC_E_DEVICE;
        if (runs) {
        st = mscan(d, (const gc_i32 *)cnt_i, off_i, pcap, totals + 1, s);
        if (st != GC_OK) return st;
        st = mscan(d, (const gc_i32 *)cnt_j, off_j, pcap, totals + 2, s);
        if (st != GC_OK) return st;

        st = scratch(d, GCN_SC_BITS, pcap * (gc_i64)sizeof(gc_i64), s, presized, (void **)&bits);
        if (st) return st;
        st = scratch(d, GCN_SC_OFF_M, pcap * (gc_i64)sizeof(gc_i64), s, 0, (void **)&off_m);
        if (st) return st;
        gcn_kern_mask_count<<<(unsigned)grid_for(pcap, GCN_BLOCK), GCN_BLOCK,
                              0, s>>>(d->pair, cnt_i, cnt_j, d->cell, bits, pcap,
                                      np_dev, sten, report);
        st = mscan(d, (const gc_i64 *)bits, off_m, pcap, totals + 3, s);
        if (st != GC_OK) return st;
        gcn_kern_list_stats2<<<(unsigned)grid_for(pcap, GCN_BLOCK), GCN_BLOCK,
                               0, s>>>(bits, cnt_i, cnt_j, pcap, np_dev, report);
        }
        if (cudaMemcpyAsync(pin + 8, totals, 4 * sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaMemcpyAsync(pin + 16, report, 16 * sizeof(gc_i64),
                            cudaMemcpyDeviceToHost, s) != cudaSuccess)
            return GC_E_DEVICE;
        st = host_run();
        if (st == GC_OK) st = dev_phase(s, "sel_count");
        if (st != GC_OK) return st;
        if (!presized) break;
        st = candidate_verdict(ctx, d, pin, defer_classify);
        if (st != GC_OK) return st;
        npair = pin[4];
        max_run = (gc_i32)pin[5];
        if (npair <= d->pair_capacity && (max_run + 31) / 32 <= sel_words) break;
        presized = 0;    /* outgrown: size from this build's counts, count again */
    }
    d->rigid_counted = rigid_early;   /* landed with the synchronisations above */
    d->num_pairs = npair;
    const gc_i64 tot_i = pin[9], tot_j = pin[10], tot_bits = pin[11];
    const gc_i64 *rep0 = pin + 16;       /* report[0..15] as of the sizes */

    {
        gc_i64 stats[5] = { rep0[8], rep0[9], rep0[10], pin[7], (gc_i64)max_run };

        if (stats[3] != d->num_resident) {
            std::fprintf(stderr,
                "GPU_Core_Error> phase=cell_runs the cell runs hold %lld "
                "slots, the context has %lld residents; largest cell %lld over "
                "%lld cells\n",
                (long long)stats[3], (long long)d->num_resident,
                (long long)stats[4], (long long)d->ncell);
            return GC_E_CAPACITY;
        }
        if (rep0[15] != 0) {
            std::fprintf(stderr,
                "GPU_Core_Error> phase=sel_count %lld pair(s) with a run "
                "longer than the %d-atom longest cell\n",
                (long long)rep0[15], (int)max_run);
            return GC_E_CAPACITY;
        }
        if (rep0[0] != 0) {
            std::fprintf(stderr,
                "GPU_Core_Error> phase=mask_count %lld pair(s) hold a run "
                "longer than their cell; first pair %lld ni=%lld nj=%lld\n",
                (long long)rep0[0], (long long)rep0[1],
                (long long)rep0[2], (long long)rep0[3]);
            return GC_E_CAPACITY;
        }

        /* A mask that does not fit a sane fraction of the device is a refusal
         * with its numbers. Asked only when the mask has to grow; the
         * cluster-only list builds none. */
        const gc_i64 words_need = (tot_bits + 63) / 64 + 1;
        gc_i64 freeb = 0, totb = 0;
        const char *dn = 0;
        if (runs && words_need > d->mask_words &&
            device_meminfo(&freeb, &totb, &dn) == GC_OK && freeb > 0) {
            const gc_i64 want = (tot_bits + 63) / 64 * (gc_i64)sizeof(gc_u64);
            if (want > freeb / 4) {
                std::fprintf(stderr,
                    "GPU_Core_Error> phase=mask_size the exclusion mask "
                    "would need %lld MB of the %lld MB free; %lld pairs "
                    "carry a mask, longest runs %lld by %lld\n",
                    (long long)(want >> 20), (long long)(freeb >> 20),
                    (long long)stats[0], (long long)stats[1],
                    (long long)stats[2]);
                return GC_E_NOMEM;
            }
        }
    }

    {
        gc_i64 need = tot_i + tot_j;
        if (tot_i < 0 || tot_j < 0 || need > GC_MAX_LOCAL_INDEX) {
            std::fprintf(stderr, "GPU_Core_Error> phase=rebuild_list "
                         "selection=%lld exceeds the 32-bit pair offsets\n",
                         (long long)need);
            return GC_E_CAPACITY;
        }
        if (need > d->sel_capacity) {
            dev_release((void **)&d->sel);
            d->sel_capacity = need + need / 8 + 64;
            st = dev_calloc((void **)&d->sel,
                            d->sel_capacity * (gc_i64)sizeof(gc_i32));
            if (st) return st;
        }
    }
    /* the j runs follow the i runs in one flat array, so the j offsets
       are shifted past the i block rather than kept in a second array */
    if (runs) {
    gcn_kern_sel_fill<<<pair_grid(npair), GCN_BLOCK, 0, s>>>(
        d->cell, d->resident_slot, d->pair,
        off_i, off_j, cnt_i, cnt_j, hit, sel_words, place, d->sel, npair, tot_i);
    {
        size_t tb = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, tb, (const gc_u8 *)0, (gc_u8 *)0,
                                        (const gc_i32 *)0, d->pair_order,
                                        (int)npair, 0, 8, s);
        const gc_i64 tbr = ((gc_i64)tb + 255) / 256 * 256;
        const gc_i64 vbr = (npair * (gc_i64)sizeof(gc_i32) + 255) / 256 * 256;
        char *w = 0;
        st = scratch(d, GCN_SC_PORDER, tbr + vbr + 2 * npair, s, 0, (void **)&w);
        if (st) return st;
        gc_i32 *idx = (gc_i32 *)(w + tbr);
        gc_u8 *kin = (gc_u8 *)(w + tbr + vbr), *kout = kin + npair;
        gcn_kern_pair_load<<<(unsigned)grid_for(npair, GCN_BLOCK), GCN_BLOCK,
                             0, s>>>(cnt_i, cnt_j, npair, kin, idx);
        if (cub::DeviceRadixSort::SortPairs(w, tb, kin, kout, idx, d->pair_order,
                                            (int)npair, 0, 8, s) != cudaSuccess)
            return GC_E_DEVICE;
    }
    }

    st = rb_phase(s, "sel_fill");
    if (st != GC_OK) return st;

    if (runs) {
    gc_i64 words = (tot_bits + 63) / 64 + 1;
    if (words > d->mask_words) {
        dev_release((void **)&d->mask);
        d->mask_words = words + words / 8 + 8;
        st = dev_calloc((void **)&d->mask,
                        d->mask_words * (gc_i64)sizeof(gc_u64));
        if (st) return st;
    }
    gcn_kern_mask_fill<<<(unsigned)grid_for(words, GCN_BLOCK),
                         GCN_BLOCK, 0, s>>>(d->mask, words, tot_bits);
    gcn_kern_mask_set<<<pair_grid(npair), GCN_BLOCK, 0, s>>>(
        d->pair, bits, off_m, d->sel, d->mask, npair, d->mask_words,
        d->sel_capacity, report);

    st = rb_phase(s, "mask_set");
    if (st != GC_OK) return st;
    }

    st = host_drain();
    if (st != GC_OK) return st;

    {
        gc_i64 cap = 1;
        while (cap < 2 * d->num_resident) cap <<= 1;
        gc_gid *mkey = 0; gc_image *mikey = 0; gc_i32 *mval = 0;
        const int ncel_x = d->cell_count[0], ncel_y = d->cell_count[1],
                  ncel_z = d->cell_count[2];
        int2 *pk = 0;
        if (carry) {
            st = scratch(d, GCN_SC_CARRYMAP, na * (gc_i64)sizeof(int2), s, 0,
                         (void **)&pk);
            if (st != GC_OK) return st;
            gcn_kern_carry_map<<<(unsigned)grid_for(na, GCN_BLOCK), GCN_BLOCK,
                                 0, s>>>(inv, na, d->cell_of, d->cell, pk,
                                         flags + 2);
        }
        const int audit = 0;
        if (!carry || audit) {
        st = scratch(d, GCN_SC_MKEY, cap * (gc_i64)sizeof(gc_gid), s, 0, (void **)&mkey);
        if (st) return st;
        st = scratch(d, GCN_SC_MIKEY, cap * (gc_i64)sizeof(gc_image), s, 0, (void **)&mikey);
        if (st) return st;
        st = scratch(d, GCN_SC_MVAL, cap * (gc_i64)sizeof(gc_i32), s, 0, (void **)&mval);
        if (st) return st;
        gcn_kern_imap_clear<<<(unsigned)grid_for(cap, GCN_BLOCK), GCN_BLOCK,
                              0, s>>>(mkey, mikey, mval, cap);
        cudaMemsetAsync(report + 13, 0, sizeof(gc_i64), s);
        gcn_kern_imap_insert<<<(unsigned)grid_for(d->num_resident, GCN_BLOCK),
                               GCN_BLOCK, 0, s>>>(d->gid, d->image,
            d->num_resident, mkey, mikey, mval, cap, report + 13);
        st = rb_phase(s, "identity_map");
        if (st != GC_OK) return st;
        }
        cudaMemsetAsync(totals + 1, 0, 2 * sizeof(gc_i64), s);

        /* The canonical endpoint records are resident (term_copy_sync):
         * uploaded when the host pools changed, else reused; every kind is
         * launched before the one synchronisation. */
        const int reuse_terms = 1;
        gcn_rebuild_cache &rc = rebuild_cache(d);

        /* The final stock real-space zero bits include neutral 1-2/1-3 pairs
         * absent from the charge-gated reciprocal correction list; resolve
         * them with the same mask clearer. No force term is added. */
        if (!rc.mask_dev && (ctx->real_mask_gid.size() & 1u)) return GC_E_STATE;
        static_assert(sizeof(struct gcn_term_exec) == 64,
                      "real-mask peak plan assumes 64-byte execution records");
        const gc_i64 nm = rc.mask_dev ? rc.real_mask.nt :
                          (gc_i64)(ctx->real_mask_gid.size() / 2);
        struct gcn_term_exec *ex = 0;
        if (nm > 0) {
            if (!rc.mask_dev && (gc_i64)ctx->real_mask_image.size() != 2 * nm)
                return GC_E_ENDPOINT;
            st = scratch(d, GCN_SC_RM_EX,
                         nm * (gc_i64)sizeof(struct gcn_term_exec), s, 0,
                         (void **)&ex);
            if (st != GC_OK) return st;
        }

        struct gcn_mask_args ma;
        ma.cell_of = d->cell_of; ma.cell = d->cell; ma.pair = d->pair;
        ma.pair_at = pair_at; ma.sel = runs ? d->sel : 0;
        ma.resident_slot = d->resident_slot; ma.hit = hit; ma.words = sel_words;
        ma.place = place; ma.mask = runs ? d->mask : 0; ma.bad = totals + 2;
        ma.mask_words = d->mask_words; ma.st = sten;
        ma.xl = 0; ma.xn = 0; ma.xcap = 0;
        struct gcn_term_segs mseg;
        mseg.n = 0;
        gcn_seg_add(mseg, ex, nm, GC_TERM_EXCL, 2, 1, GCN_MASK_POOL_REAL);
        for (int k = GC_TERM_NB14; k <= GC_TERM_EXCL; ++k)
            gcn_seg_add(mseg, d->term[k], d->term_count[k], k, gc_term_arity[k],
                        gc_term_pbc_codes[k] > 0 ? 0 : 1,
                        (k == GC_TERM_NB14) ? GCN_MASK_POOL_NB14
                                            : GCN_MASK_POOL_EXCL);
        st = native_nbc_excl_list(d, mseg.n > 0 ? mseg.start[mseg.n] : 0, s,
                                  &ma.xl, &ma.xn, &ma.xcap);
        if (st != GC_OK) { scratch_drop((void **)&ex); return st; }

        if (carry) {
            /* one launch carries every kind's records and the real mask's */
            struct gcn_term_segs seg;
            seg.n = 0;
            for (int k = 0; k < GC_TERM_NB14; ++k)
                gcn_seg_add(seg, d->term[k], d->term_count[k], k,
                            gc_term_arity[k], gc_term_pbc_codes[k] > 0 ? 0 : 1,
                            -1);
            for (int k = 0; k < mseg.n; ++k)
                gcn_seg_add(seg, mseg.rec[k],
                            mseg.start[k + 1] - mseg.start[k], mseg.kind[k],
                            mseg.arity[k], mseg.codes[k], mseg.pool[k]);
            if (seg.n > 0) {
                gcn_kern_term_remap<<<(unsigned)grid_for(seg.start[seg.n],
                                                         GCN_BLOCK),
                                      GCN_BLOCK, 0, s>>>(
                    pk, na, seg, ncel_x, ncel_y, ncel_z, totals + 1, ma);
                st = dev_launched("term_remap");
                if (st != GC_OK) { scratch_drop((void **)&ex); return st; }
            }
        }
        for (int k = 0; k < GC_TERM_NKIND; ++k) {
            gc_i64 nt = d->term_count[k];
            if (nt <= 0) continue;
            if (carry) {
                if (audit) {
                    gcn_term_copy &tc = rc.term[k];
                    const gc_i32 ncodes = gc_term_pbc_codes[k];
                    struct gcn_term_exec *ref = 0;
                    st = scratch(d, GCN_SC_CARRYCHK,
                                 nt * (gc_i64)sizeof(struct gcn_term_exec), s, 0,
                                 (void **)&ref);
                    if (st != GC_OK) return st;
                    st = term_pool_from_host(ctx, rc, k, nt, reuse_terms, s);
                    if (st != GC_OK) return st;
                    gcn_kern_term_exec<<<(unsigned)grid_for(nt, GCN_BLOCK), GCN_BLOCK,
                                         0, s>>>(
                        tc.ep, tc.epi, tc.pid, ncodes > 0 ? tc.pbc : 0, nt,
                        gc_term_arity[k], mkey, mikey, mval, cap,
                        d->cell, d->cell_of, ref, totals + 1,
                        d->cell_count[0], d->cell_count[1], d->cell_count[2], k,
                        ncodes);
                    gcn_kern_exec_compare<<<(unsigned)grid_for(nt, GCN_BLOCK),
                                            GCN_BLOCK, 0, s>>>(
                        ref, d->term[k], nt, flags + 3);
                    if (cudaMemcpyAsync(pin + 57, flags + 3, sizeof(gc_i64),
                                        cudaMemcpyDeviceToHost, s) != cudaSuccess)
                        return GC_E_DEVICE;
                    st = dev_phase(s, "carry_verify");
                    if (st != GC_OK) return st;
                    if (ctx->rank == 0)
                        std::fprintf(stdout,
                            "GPU_Core_Check> carry_verify kind=%d records=%lld "
                            "differ=%lld\n", k, (long long)nt, (long long)pin[57]);
                    if (pin[57] != 0) return GC_E_MISMATCH;
                }
                continue;
            }
            const gc_i32 ncodes = gc_term_pbc_codes[k];
            gcn_term_copy &tc = rc.term[k];
            if (!rc.term_dev) {
                st = term_pool_from_host(ctx, rc, k, nt, reuse_terms, s);
                if (st != GC_OK) return st;
            }
            gcn_kern_term_exec<<<(unsigned)grid_for(nt, GCN_BLOCK), GCN_BLOCK,
                                 0, s>>>(
                tc.ep, tc.epi, tc.pid, ncodes > 0 ? tc.pbc : 0, nt,
                gc_term_arity[k], mkey, mikey, mval, cap,
                d->cell, d->cell_of, d->term[k], totals + 1,
                d->cell_count[0], d->cell_count[1], d->cell_count[2], k,
                ncodes);
            st = dev_launched("term_exec");
            if (st != GC_OK) return st;
        }

        if (!carry) {
            if (nm > 0) {
                gcn_term_copy &mc = rc.real_mask;
                if (!rc.mask_dev)
                    st = term_copy_sync(mc, reuse_terms, nm, 2 * nm, 0,
                                        &ctx->real_mask_gid[0],
                                        &ctx->real_mask_image[0], 0, 0, 0, s);
                if (st != GC_OK) { scratch_drop((void **)&ex); return st; }
                gcn_kern_term_exec<<<(unsigned)grid_for(nm, GCN_BLOCK),
                                     GCN_BLOCK, 0, s>>>(
                    mc.ep, mc.epi, 0, 0, nm, 2, mkey, mikey, mval, cap,
                    d->cell, d->cell_of, ex, totals + 1,
                    d->cell_count[0], d->cell_count[1], d->cell_count[2],
                    GC_TERM_EXCL, 0);
                st = dev_launched("real_mask_resolve");
            }
            if (st == GC_OK && mseg.n > 0) {
                gcn_kern_mask_exclusions<<<(unsigned)grid_for(mseg.start[mseg.n],
                                                              GCN_BLOCK),
                                           GCN_BLOCK, 0, s>>>(mseg, ma);
                st = dev_launched("mask_exclusions");
            }
        }
        scratch_drop((void **)&ex);
        if (st != GC_OK) return st;

        /* After the setup's build, the term stage's traps join the step's
         * deferred checks, read at the next synchronising call before
         * anything computed from this list is published; an unplaceable
         * system declines at setup. */
        const int defer_traps = d->scheduled_rebuilds + d->early_rebuilds > 0;
        if (defer_traps) {
            gcn_kern_list_verdict<<<1, 1, 0, s>>>(totals, report, flags + 2,
                                                  d->verdict);
            st = dev_launched("list_verdict");
            scratch_drop((void **)&mkey);
            scratch_drop((void **)&mikey);
            scratch_drop((void **)&mval);
        } else {
            if (cudaMemcpyAsync(pin + 32, totals, 4 * sizeof(gc_i64),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaMemcpyAsync(pin + 40, report, 16 * sizeof(gc_i64),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess ||
                cudaMemcpyAsync(pin + 56, flags + 2, sizeof(gc_i64),
                                cudaMemcpyDeviceToHost, s) != cudaSuccess)
                return GC_E_DEVICE;
            st = dev_phase(s, "mask_exclusions");
            if (st != GC_OK) return st;
            {
                const gc_i64 *rep1 = pin + 40;
                if (rep1[4] != 0) {
                    std::fprintf(stderr,
                        "GPU_Core_Error> phase=mask_set %lld pair(s) index past the "
                        "mask; first pair %lld bits=%lld off=%lld capacity=%lld "
                        "words, sel=%lld\n",
                        (long long)rep1[4], (long long)rep1[5], (long long)rep1[6],
                        (long long)rep1[7], (long long)d->mask_words,
                        (long long)d->sel_capacity);
                    return GC_E_CAPACITY;
                }
                if (rep1[13] != 0) return GC_E_MISMATCH;   /* identity map */
                if (pin[56] != 0) {                          /* the carry's perm */
                    std::fprintf(stderr,
                        "GPU_Core_Error> phase=term_remap %lld permutation "
                        "entries outside the owned slots\n", (long long)pin[56]);
                    return GC_E_STATE;
                }
            }
            gc_i64 bad[4] = { pin[32], pin[33], pin[34], pin[35] };
            scratch_drop((void **)&mkey);
            scratch_drop((void **)&mikey);
            scratch_drop((void **)&mval);
            if (bad[1] != 0) { st = GC_E_ENDPOINT; }
            else if (bad[2] != 0) {
                std::fprintf(stderr, "GPU_Core_Error> phase=mask_exclusions rank=%d "
                             "unplaced_terms=%lld\n", ctx->rank, (long long)bad[2]);
                st = GC_E_UNSUPPORTED;
            }
            else st = GC_OK;
        }
        if (st != GC_OK) {
            scratch_drop((void **)&keep);    scratch_drop((void **)&pair_at);
            scratch_drop((void **)&slot);    scratch_drop((void **)&cnt_i);
            scratch_drop((void **)&cnt_j);   scratch_drop((void **)&off_i);
            scratch_drop((void **)&off_j);   scratch_drop((void **)&bits);
            scratch_drop((void **)&off_m);   scratch_drop((void **)&totals);
            scratch_drop((void **)&report);
            return st;
        }
    }

    scratch_drop((void **)&keep);    scratch_drop((void **)&pair_at);
    scratch_drop((void **)&slot);    scratch_drop((void **)&cnt_i);
    scratch_drop((void **)&cnt_j);   scratch_drop((void **)&off_i);
    scratch_drop((void **)&off_j);   scratch_drop((void **)&bits);
    scratch_drop((void **)&off_m);   scratch_drop((void **)&totals);
    scratch_drop((void **)&report);

    st = rb_phase(s, "rebuild_list");
    if (st == GC_OK) st = native_term_pairs(d, s);
    if (st != GC_OK) return st;
    if (ctx->nproc == 1 && !migrated && d->num_resident == d->num_owned) {
        gcn_rebuild_cache &rc = rebuild_cache(d);
        bool all = true;
        for (int k = 0; k < GC_TERM_NKIND; ++k) {
            rc.exec_nt[k] = d->term_count[k];
            if (d->term_count[k] > 0 && !rc.term[k].valid) all = false;
        }
        rc.exec_nm = (gc_i64)(ctx->real_mask_gid.size() / 2);
        if (rc.exec_nm > 0 && !rc.real_mask.valid) all = false;
        rc.exec_rm = rc.sc[GCN_SC_RM_EX];
        rc.exec_nres = d->num_resident;
        rc.exec_ok = all;
    }

    /* After a migration the context's counts follow the device. The host
     * mirror's per-atom arrays are not resynchronised (the next transaction
     * resolves against the device). */
    if (migrated) {
        ctx->num_owned = d->num_owned;
        ctx->num_groups = d->num_groups;
        ctx->num_group_members = d->num_members;
    }
    return GC_OK;
}

gc_status native_rebuild(gc_context *ctx, gc_i32 early)
{
    return native_rebuild_impl(ctx, early, 1);
}

/* ---- the guard ----------------------------------------------------- */

gc_status native_list_guard_launch(gc_context *ctx, gc_f64 over,
                                   cudaStream_t s)
{
    struct gcn_device *d = ctx->native;
    /* GCN_MAX_BLOCKS partial sums at most (reduce_partial), in a fixed
       order: the guard's grid is not grid_for's */
    gc_i64 nb = (d->num_owned + GCN_RED_BLOCK - 1) / GCN_RED_BLOCK;
    if (nb > GCN_MAX_BLOCKS) nb = GCN_MAX_BLOCKS;
    if (nb < 1) nb = 1;

    /* VV1's solver passes took the distances: finalize and clear them */
    if (d->guard_ready) {
        d->guard_ready = 0;
        gcn_kern_guard_finalize<<<1, GCN_RED_BLOCK, 0, s>>>(
            d->guard_max, 1, d->guard_pin, d->guard_idx, d->guard_done,
            over, d->guard_max);
        return dev_launched("list_guard");
    }
    gcn_kern_displacement<<<(unsigned)nb, GCN_RED_BLOCK, 0, s>>>(
        d->coord, d->list_ref, d->coord_ref, d->num_owned, d->pitch,
        d->box[0], d->box[1], d->box[2], d->reduce_partial);
    gcn_kern_guard_finalize<<<1, GCN_RED_BLOCK, 0, s>>>(
        d->reduce_partial, (int)nb, d->guard_pin, d->guard_idx,
        d->guard_done, over, (gc_f64 *)0);
    return dev_launched("list_guard");
}

}  /* namespace gcn */
