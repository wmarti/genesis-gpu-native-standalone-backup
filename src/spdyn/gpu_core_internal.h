/* gpu_core_internal.h : private types of the device-native core (not part of the ABI). */

#ifndef GPU_CORE_INTERNAL_H
#define GPU_CORE_INTERNAL_H

#include "gpu_core_abi.h"
#include "gpu_core_xchg.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace gcn {

/* An LSD radix sort, 11 bits a pass and as many passes as the largest key
 * needs, into the order std::stable_sort by that key gives.  A key below
 * zero leaves v unchanged and returns false (the caller sorts otherwise). */
template <class T, class Key>
bool radix_sort(std::vector<T> &v, Key key)
{
    gc_i64 top = 0;
    for (const T &x : v) {
        const gc_i64 k = (gc_i64)key(x);
        if (k < 0) return false;
        if (k > top) top = k;
    }
    if (v.size() < 256) {
        std::stable_sort(v.begin(), v.end(), [&](const T &a, const T &b) {
            return (gc_i64)key(a) < (gc_i64)key(b);
        });
        return true;
    }
    std::vector<T> tmp(v.size());
    for (int sh = 0; sh < 63 && (top >> sh) != 0; sh += 11) {
        size_t cnt[2049] = { 0 };
        for (const T &x : v) ++cnt[((gc_u64)key(x) >> sh & 2047) + 1];
        for (int b = 0; b < 2048; ++b) cnt[b + 1] += cnt[b];
        for (const T &x : v) tmp[cnt[(gc_u64)key(x) >> sh & 2047]++] = x;
        v.swap(tmp);
    }
    return true;
}

/* Checked integer arithmetic (doc/21_GPU_Native.rst). */

bool mul_checked(gc_i64 a, gc_i64 b, gc_i64 *out);
bool add_checked(gc_i64 a, gc_i64 b, gc_i64 *out);
/* Narrow a 64-bit count to an admitted 32-bit local index.  False means
 * the value did not fit; the caller returns GC_E_OVERFLOW. */
bool narrow(gc_i64 v, gc_i32 *out);

/* Open addressing, power-of-two capacity, linear probing, sized at twice the
 * entry count (O(local cells or atoms)). */

class KeyMap {
  public:
    KeyMap() : mask_(0) {}
    void reset(gc_i64 entries);
    /* Returns false when the key is already present. */
    bool insert(gc_i64 key, gc_i32 value);
    /* Returns -1 when absent. */
    gc_i32 find(gc_i64 key) const;
    gc_i64 bytes() const;
  private:
    /* Empty marker: keys are non-negative, so the most negative value is never real. */
    static gc_i64 empty() { return (gc_i64)(-9223372036854775807LL) - 1; }
    std::vector<gc_i64> key_;
    std::vector<gc_i32> val_;
    gc_i64 mask_;
};

/* Exact local atom identity.  Owned images are zero; ghosts keep the
 * packed periodic-copy image captured by GENESIS. */
struct AtomIdentity { gc_gid gid; gc_image image; gc_i32 slot; };

class AtomIdentityMap {
  public:
    gc_status build(const std::vector<gc_gid> &owned,
                    const std::vector<gc_gid> &ghost_gid,
                    const std::vector<gc_image> &ghost_image);
    gc_status find_exact(gc_gid gid, gc_image image, gc_i32 *slot) const;
    gc_status find_unique(gc_gid gid, gc_i32 *slot) const;
    gc_i64 bytes() const;
  private:
    std::vector<AtomIdentity> entries_;
};


struct TermPool {
    gc_i32 kind;
    gc_i32 arity;
    gc_i64 count;
    std::vector<gc_gid>     endpoint;      /* arity * count, GIDs          */
    std::vector<gc_image>   endpoint_image;/* selected periodic-copy image */
    std::vector<gc_u8>      endpoint_image_valid;
    std::vector<gc_term_id> term_id;       /* count                        */
    std::vector<gc_i32>     param_id;      /* count                        */
    std::vector<gc_i32>     image;         /* count                        */
    std::vector<gc_i32>     pbc;           /* stock codes, code first     */
    std::vector<gc_i32>     genesis_cell;  /* count, cell GENESIS listed it */
    std::vector<gc_i32>     owner_cell;    /* count, -1 = not resolved     */
    std::vector<gc_u8>      owned_here;    /* count                        */
    /* deduplicated parameter payloads, param_stride doubles each */
    gc_i64                  param_stride;
    std::vector<gc_f64>     param_payload;
    gc_i64                  param_count;

    TermPool() : kind(0), arity(0), count(0), param_stride(0), param_count(0) {}
    void clear();
    gc_i64 bytes() const;
    gc_i64 execution_bytes() const;
    gc_status resolve(const AtomIdentityMap &map, size_t at, gc_i32 *slot) const;
};

/* Implemented in gpu_core_device.cu. */
gc_status device_meminfo(gc_i64 *bytes_free, gc_i64 *bytes_total,
                         const char **name_out);

/* Declared without CUDA types so host-compiled gpu_core.cpp can attach and release it. */

gc_status native_context_attach(gc_context *ctx);
gc_status native_context_release(gc_context *ctx);
/* Positional restraints, the whole gathered table: gid[n] and par7[7n]
 * (k, w_x, w_y, w_z, ref_x, ref_y, ref_z).  Implemented in gpu_force.cu. */
gc_status native_posres_upload(gc_context *ctx, gc_i64 n, const gc_gid *gid,
                               const gc_f64 *par7);
/* Host import must join rank-wide admission before device route setup. */
gc_status native_dist_vote(gc_context *ctx, gc_status local, const char *phase);
/* Every rank's words, concatenated in rank order (setup-time only). */
gc_status native_dist_gather(gc_context *ctx, const std::vector<gc_i64> &mine,
                             std::vector<gc_i64> *all);

/* a - b per axis of two packed images (three int16 cell shifts). */
inline gc_image image_sub(gc_image a, gc_image b)
{
    gc_image r = 0;
    for (int k = 0; k < 3; ++k) {
        const gc_i32 d = (gc_i32)(gc_i16)((a >> (16 * k)) & 0xffffLL) -
                         (gc_i32)(gc_i16)((b >> (16 * k)) & 0xffffLL);
        r |= ((gc_image)(gc_i16)d & 0xffffLL) << (16 * k);
    }
    return r;
}

/* The list-validity guard (doc/21_GPU_Native.rst): list radius, the support
 * radius the tables reach (max(table, prune)) and the half skin. */
struct GuardPlan {
    gc_f64 pairlistdist;
    gc_f64 support_radius;
    gc_f64 half_skin;
};

/* One hydrogen group's constraint row: its representative GID, H count
 * and bond distances (stock HGr_bond_dist); it travels with the group. */
struct RigidRow {
    gc_gid gid;
    gc_i32 arity;
    gc_i32 pad0;
    gc_f64 dist[GC_MAX_HGROUP_H];
};

}  /* namespace gcn */


struct gc_context_s {
    gc_i32 rank;
    gc_i32 nproc;
    gc_i32 replica;
    gc_i32 comm;
    gc_epoch epoch;

    gc_geometry_desc geo;

    /* cells: local first, then boundary */
    gc_i64 ncell_local;
    gc_i64 ncell_boundary;
    gc_i64 ncell;
    std::vector<gc_i32>     cell_gx, cell_gy, cell_gz;   /* zero based     */
    std::vector<gc_cell_key> cell_key;
    std::vector<gc_f64>     cell_tie;
    gcn::KeyMap             cell_map;                    /* key -> index   */

    /* owned atoms, grouped SoA, no padding, sorted by
     * (cell_key, group_kind, group_gid) with members contiguous */
    gc_i64 num_owned;
    std::vector<gc_gid> atom_gid;
    std::vector<gc_f64> atom_coord;      /* 3 * n */
    std::vector<gc_f64> atom_vel;        /* 3 * n */
    std::vector<gc_f64> atom_charge;
    std::vector<gc_f64> atom_mass;
    std::vector<gc_f64> atom_inv_mass;
    std::vector<gc_i32> atom_cls;
    std::vector<gc_i32> atom_cell;       /* local cell index               */
    std::vector<gc_i32> atom_group;      /* group index                    */

    /* ghosts: (GID, image) identity, coordinates and charge only */
    gc_i64 num_ghost;
    std::vector<gc_gid> ghost_gid;
    std::vector<gc_image> ghost_image;
    std::vector<gc_f64> ghost_coord;     /* 3 * n */
    std::vector<gc_f64> ghost_charge;
    std::vector<gc_i32> ghost_cls;
    std::vector<gc_i32> ghost_cell;

    /* Halo admission: exact atom counts per pass (axis) and edge (lower, upper),
     * agreed with neighbours before device storage is allocated. */
    gc_i64 native_outer_send[3][2];
    gc_i64 native_outer_recv[3][2];

    /* GID -> index: owned atoms in [0, num_owned), ghosts encoded as
     * -(ghost index) - 1.  Sparse, O(owned + ghosts). */
    gcn::KeyMap gid_map;
    gcn::AtomIdentityMap identity_map;

    /* groups over owned atoms */
    gc_i64 num_groups;
    gc_i64 num_group_members;
    std::vector<gc_i64> group_offset;    /* num_groups + 1                 */
    std::vector<gc_i32> group_member;    /* local atom indices             */
    std::vector<gc_i32> group_rep;
    std::vector<gc_u8>  group_kind;
    std::vector<gc_gid> group_gid;
    std::vector<gc_i32> group_cell;

    gcn::TermPool term[GC_TERM_NKIND];
    std::vector<gc_gid> real_mask_gid; /* unordered stock zero-bit pairs, 2 per pair */
    std::vector<gc_image> real_mask_image; /* exact image of each mask endpoint */
    gc_i32 real_mask_ready;
    /* positional restraints the device table holds (gpu_core_posres_set) */
    gc_i64 posres_count;
    /* Checked build: host mirror of the device rigid table
     * (gcn_device::rigid_*), sorted by representative GID, which migration
     * edits and compares with the device's. */
    std::vector<gcn::RigidRow> rigid_rows;

    gcn::GuardPlan guard;

    /* Device-resident step state (gpu_core_native.h); opaque here. */
    struct gcn_device *native;


    gc_context_s();
};


#endif /* GPU_CORE_INTERNAL_H */
