/*
 * gpu_core_native.h : the device-resident state of the native step.
 *
 * Private to the CUDA units; nothing here crosses the C ABI (see
 * doc/21_GPU_Native.rst). One slot per owned atom, ordered by the rebuild's
 * stable group sort; component c of slot s is at [c*pitch + s]. Coordinates
 * are in this rank's frame, each cell pair carrying its own periodic offset.
 * Real-space, bonded and reciprocal forces are accumulated separately and
 * joined once per evaluation because their virial conventions differ. State is
 * FP64, except that nonbond_precision = MIXED stores velocities and the joined
 * force in FP32 (gcn_vf).
 */

#ifndef GPU_CORE_NATIVE_H
#define GPU_CORE_NATIVE_H

#include "gpu_core_abi.h"
#include "gpu_core_internal.h"
#include "gpu_domain.h"

#include <cuda_runtime.h>
#include <functional>

/* Real-space kernel warp shape: 8 i-atom groups by 4 cached j slabs. The
 * reduction order depends on it, so it is a constant, not a tuning knob. */
#define GCN_RS_IX       8
#define GCN_RS_IY       4
#define GCN_RS_WARPS    4
#define GCN_RS_BLOCK    (32 * GCN_RS_WARPS)
/* j atoms staged per warp in shared memory (a max_iy_natom slab). */
#define GCN_RS_JBLOCK   56

#define GCN_BLOCK       128
#define GCN_RED_BLOCK   256
/* Candidate stencil half width in cells; the rebuild refuses a list radius
 * needing more rather than miss pairs. */
#define GCN_MAX_STENCIL 4

/* Bonded energy/virial accumulator slots: the first six are gc_energy_slot's
 * bonded entries in order; the last three are the diagonal virial in the
 * sum-of-d*w convention that `virial = virial - viri` subtracts. */
#define GCN_BE_BOND     0
#define GCN_BE_ANGLE    1
#define GCN_BE_UREY     2
#define GCN_BE_DIHE     3
#define GCN_BE_IMPR     4
#define GCN_BE_CMAP     5
#define GCN_BE_ELEC14   6
#define GCN_BE_VDW14    7
#define GCN_BE_ELECCOR  8
#define GCN_BE_VIRX     9
#define GCN_BE_VIRY    10
#define GCN_BE_VIRZ    11
/* positional restraints: energy, then GENESIS's virial_ext diagonal
 * (-x_d * w_d, its own sign; it is not part of the internal virial) */
#define GCN_BE_POSRES  12
#define GCN_BE_PVIRX   13
#define GCN_BE_PVIRY   14
#define GCN_BE_PVIRZ   15
#define GCN_BE_NSLOT   16

#define GCN_RE_ELEC     0
#define GCN_RE_VDW      1
#define GCN_RE_VIRX     2
#define GCN_RE_VIRY     3
#define GCN_RE_VIRZ     4
#define GCN_RE_NSLOT    5

#define GCN_PE_ENE      0
#define GCN_PE_VIRX     1
#define GCN_PE_VIRY     2
#define GCN_PE_VIRZ     3
#define GCN_PE_NSLOT    4

struct gcn_cell {
    gc_i32 atom_begin;
    gc_i32 atom_count;
    gc_i32 group_begin;
    gc_i32 group_count;
    gc_i32 gx, gy, gz;          /* extended, zero based                  */
    gc_i32 pad0;
};

/* One cell pair, self contained: every quantity the kernel needs is here or
 * in the runs it names.
 *   ix_off/ix_n, iy_off/iy_n  index gcn_device::sel (selected ABSOLUTE slots);
 *   mask_off  bit offset into gcn_device::mask; bit (mask_off + a*iy_n + b) is
 *             1 when pair (ix[a], iy[b]) is evaluated; -1 = every pair active;
 *   move      whole-box periodic offset added to the i side, in box lengths;
 *   virial    1 when move is not zero (GENESIS's virial_check). */
struct gcn_pair {
    gc_i32 ci, cj;
    gc_i32 ix_off, ix_n;
    gc_i32 iy_off, iy_n;
    gc_i64 mask_off;
    gc_f64 move[3];
    gc_i32 virial;
    gc_i32 self_pair;           /* 1 when ci == cj                       */
};

/* One execution record of a bonded term: endpoint slots and the periodic
 * image codes pbc_decode expects. Six codes because CMAP is two dihedrals
 * over eight atoms; other kinds use the first one to three and leave the rest
 * at the identity code 13. */
struct gcn_term_exec {
    gc_i32 a[GC_MAX_TERM_ARITY];
    gc_i32 pbc[6];
    gc_i32 param;               /* index into the kind's parameter table */
    gc_i32 pad0;
};

/* The fields a two-endpoint kind's kernel reads (bond, 1-4, excluded): 16
 * bytes per term instead of 64 (native_term_pairs, after each rebuild). */
struct alignas(16) gcn_term_pair {
    gc_i32 a0, a1, pbc, param;
};

/* Candidate migration payload: not an epoch, and no consumer reads it until
 * the group, canonical-term and owner votes commit. The atom record carries
 * every owned integrator field needed to resume a step on the new rank;
 * derived forces are rebuilt. */
#define GCN_MIGRATION_RECORD_SCHEMA 1
struct gcn_group_migration_wire {
    gc_i32 schema, source_rank, destination_rank, kind;
    gc_i64 epoch, atom_offset, atom_count;
    gc_gid group_gid;
    gc_i32 destination_cell[3];
    gc_i32 pad0;
};
struct gcn_atom_migration_wire {
    gc_i32 schema, source_rank, destination_rank, member_ordinal;
    gc_i64 epoch;
    gc_gid gid, group_gid;
    gc_f64 coord[3], coord_ref[3], list_ref[3];
    gc_f64 vel[3], vel_ref[3], vel_half[3], vel_full[3];
    gc_f64 force[3];
    gc_f64 charge, mass, inv_mass;
    gc_i32 cls, pad0;
};
static_assert(sizeof(gcn_group_migration_wire) == 64,
              "group migration wire size");
static_assert(sizeof(gcn_atom_migration_wire) == 264,
              "atom migration wire size");

/* The reciprocal pipeline, private to gpu_pme.cu. */
struct gcn_pme_mesh;
/* The cluster-pair real-space list, private to gpu_nbcluster.cu. */
struct gcn_nbc;

/* A velocity or joined-force field, [3*pitch]. MIXED stores FP32 elements
 * (the storage of GROMACS's mixed precision, Pall et al., J. Chem. Phys. 153,
 * 134110, 2020); arithmetic and every sum over them stay FP64 or fixed point.
 * The pointer is untyped so every reader names the element type (vf_each). */
struct gcn_vf {
    void *p;
    template <class S> S *as() const { return (S *)p; }
};

template <class F>
static inline void vf_each(int vf32, F &&f)
{
    if (vf32) f(float());
    else      f(double());
}

#define GCN_LANES 3   /* the step's side lanes beside `stream` */
/* Plain-step graph kinds (gc_graph_kind): 1 no thermostat tick, 2 a tick. */
#define GCN_GRAPH_KINDS 3
#define GCN_GRAPH_EAGER   0
#define GCN_GRAPH_CAPTURE 1
#define GCN_GRAPH_REPLAY  2

/* The device thermostat state: a tick's kinetic sums, velocity scale, NHC
 * chain and the coming Bussi draws, so a tick makes no host round trip. */
#define GCN_THERMO_DRAWS 64   /* Bussi ticks one upload covers          */
#define GCN_NHC_MAX 10        /* s_dynvars's chain arrays               */
struct gcn_thermo {
    double kin[6];            /* the tick's half and reference kinetic
                                 tensors, over every rank               */
    double scale;             /* the tick's velocity scale              */
    double nh_vel[GCN_NHC_MAX], nh_force[GCN_NHC_MAX];
    double nh_coef[GCN_NHC_MAX];
    double kin_local[6];      /* this rank's share of kin               */
    unsigned long long words[12];  /* kin_local, then the sum over the
                                 ranks, as gcn_fx fixed-point words     */
    unsigned int tick_blocks; /* blocks of the fused tick pass done   */
    unsigned int pending;     /* a tick the VV2 passes summed for but
                                 had no draw for: 1 + summed, run by
                                 the next upload (gcn_kern_tick_pending) */
    /* the upload: (rr, sum_gauss) per Bussi tick, in GENESIS's draw
       order, and how many of them are used / valid */
    double draw[2 * GCN_THERMO_DRAWS];
    gc_i32 next, count;
};

struct gcn_thermo_args {
    gc_i32 kind;              /* gc_thermostat                          */
    gc_i32 nh_length, nh_step;
    gc_i32 pad;
    double degree, kboltz, temp0;
    double dt_tau;            /* dt/tau_t        (Berendsen)            */
    double factor;            /* exp(-dt/tau_t)  (Bussi)                */
    double kbt;               /* KBOLTZ*temp0    (NHC)                  */
    double nh_dt[12];         /* dt_1, dt_2, dt_4, dt_8 per weight (NHC) */
    double nh_mass[GCN_NHC_MAX];
};

/* Stream priorities, greatest first: exchange transports, then `stream`,
 * `pme_stream` and the lanes (level 1), then `bnd_stream` over more than two
 * ranks (level 2), then the interior pairs with peers; a device with fewer
 * levels folds the lower ones onto the least above the interior pairs'. */
static inline int gcn_stream_priority(int prio_low, int prio_high, int below)
{
    int p = prio_high + below;
    if (p > prio_low - 1) p = prio_low - 1;
    return p < prio_high ? prio_high : p;
}

struct gcn_device {
    /* `stream` carries the critical path; `real_stream` the interior real-space
     * pairs, which need no halo; `pme_stream` the reciprocal sum, from the
     * step's start to the force join. Events fork and join them. */
    cudaStream_t stream;
    cudaStream_t real_stream;
    cudaEvent_t  real_done;
    cudaStream_t pme_stream;
    cudaEvent_t  pme_fork, pme_done;
    /* Side lanes beside `stream` for independent launches (bonded terms, a
     * constraint phase's two solvers): the kernels are the single-stream ones,
     * so results are bitwise identical. `aux_stream` carries the list guard's
     * measurement; with peers `bnd_stream` carries the boundary pairs. */
    cudaStream_t lane[GCN_LANES];
    cudaEvent_t  lane_fork, lane_done[GCN_LANES];
    cudaStream_t aux_stream;
    cudaEvent_t  aux_fork, aux_done;
    cudaStream_t bnd_stream;
    cudaEvent_t  bnd_fork, bnd_done;

    gc_i64 num_owned;
    gc_i64 num_ghost;
    /* nonbond_precision = MIXED: vel, vel2 and force hold FP32 elements
     * (gcn_vf), set when the tables are attached (native_vf_narrow) */
    gc_i32 vf32;
    gc_i64 num_resident;
    gc_i64 pitch;
    gc_i64 ncell;
    gc_i64 num_groups;
    gc_i64 num_members;
    gc_i64 num_pairs;
    gc_i64 pair_capacity;
    gc_i64 sel_capacity;
    gc_i64 mask_words;
    gc_i64 sort_capacity;

    /* slot capacities: migration changes the counts, so these are capacities
     * grown by native_migration_reserve before any mutation */
    gc_i64 owned_cap;
    gc_i64 group_cap;
    gc_i64 member_cap;
    gc_i64 resident_cap;
    gc_i64 sort_cap;

    gc_f64 *coord;
    gc_f64 *force_coord;        /* stock translation from list epoch     */
    gc_f64 *coord_ref;
    gc_f64 *list_ref;           /* positions when the list was built     */
    gcn_vf  vel;                /* FP32 under vf32                       */
    gc_f64 *vel_ref;
    gc_f64 *vel_half;
    gc_f64 *vel_full;
    gcn_vf  force;              /* the join, FP32 under vf32             */
    gc_f64 *force_real;
    gc_f64 *force_bond;
    gc_f64 *force_recip;
    /* force_real and force_bond as summed: single-word fixed-point
     * (gpu_fixed_sum.cuh), [c*pitch + slot], folded into the double fields
     * after their writers finish */
    unsigned long long *force_real_fx;
    unsigned long long *force_bond_fx;
    /* the (pitch, resident count) at each field's last clear: slots past
     * the resident ones stay zero while neither changes (fx_clear) */
    gc_i64 fx_clear_pitch[2], fx_clear_n[2];
    gc_f64 *charge;             /* [pitch]                               */
    gc_f64 *mass;               /* [pitch]                               */
    gc_f64 *inv_mass;           /* [pitch]                               */
    gc_i32 *cls;                /* [pitch], one based as GENESIS holds it */
    gc_gid *gid;                /* [pitch]                               */
    gc_f64 *coord2, *coord_ref2, *vel_ref2, *vel_half2, *vel_full2;
    gcn_vf  vel2;
    gc_f64 *charge2, *mass2, *inv_mass2;
    gc_i32 *cls2;
    gc_gid *gid2;
    gc_image *image;             /* [pitch], zero owned, exact ghost image */
    gc_i32 *owner;               /* [pitch], physical owner rank           */
    gc_i32 *cell_of;             /* [pitch], local box cell index          */
    gc_i32 *resident_slot;       /* [num_resident], grouped by cell        */
    gc_i64 *resident_offset;     /* [ncell+1]                              */

    struct gcn_cell *cell;
    gc_i64 *group_offset;       /* [num_groups+1]                        */
    gc_i32 *group_member;       /* [num_members], slots                  */
    gc_u8  *group_kind;         /* [num_groups]                          */
    gc_i32 *group_cell;         /* [num_groups]                          */
    gc_gid *group_gid;          /* [num_groups]                          */
    gc_i32 *group_dest_rank;    /* [num_groups], candidate owner         */
    gc_i32 *group_dest_coord;   /* [3*num_groups], global destination    */
    gc_i64 *migration_counts;   /* [3]: groups, atoms, invalid candidate */
    gc_i64 *migration_group_offset; /* [num_groups], send packet offset  */
    gc_i64 *migration_atom_offset;  /* [num_groups], send packet offset  */
    gc_i64 *migration_pack_totals;  /* [3], deterministic pack receipt   */
    gc_i64 *migration_peer_counts;  /* [3*nproc], moving term count invalid */
    gc_i64 *migration_peer_group_base; /* [nproc], packed group segments  */
    gc_i64 *migration_peer_atom_base;  /* [nproc], packed atom segments   */
    gcn_group_migration_wire *migration_group_send;
    gcn_atom_migration_wire *migration_atom_send;
    gc_f64 *group_force_move;   /* [3*num_groups], frozen at rebuild    */
    gc_i64 *group_offset2;
    gc_i32 *group_member2;
    gc_u8  *group_kind2;
    gc_i32 *group_cell2;
    gc_gid *group_gid2;
    struct gcn_pair *pair;
    gc_i32 *pair_order;         /* pairs by descending tile load: the
                                   real-space walk (gcn_kern_pair_load) */
    gc_i32 *sel;
    gc_u64 *mask;

    gc_u64 *sort_key_hi;
    gc_u64 *sort_key_lo;
    gc_i32 *sort_val;
    gc_u64 *sort_key_hi2;
    gc_u64 *sort_key_lo2;
    gc_i32 *sort_val2;
    gc_i32 *perm;               /* old slot of each new slot             */
    gc_f64 *scratch3;           /* [3*pitch] permutation staging         */

    struct gcn_term_exec *term[GC_TERM_NKIND];
    gc_i64 term_count[GC_TERM_NKIND];
    gc_i64 term_capacity[GC_TERM_NKIND];
    gc_f64 *term_param[GC_TERM_NKIND];
    gc_i64 term_param_stride[GC_TERM_NKIND];
    gc_i32 *term_param_int[GC_TERM_NKIND];
    struct gcn_term_pair *term_pair[GC_TERM_NKIND];   /* two-endpoint kinds */
    gc_i64 term_pair_n[GC_TERM_NKIND];     /* the count they were derived for */
    gc_i64 term_pair_cap[GC_TERM_NKIND];
    /* The mixed mode's excluded pairs that are not inside one rigid water
     * (excl_keep_launch, gpu_step.cu): force-only steps need no others */
    struct gcn_term_pair *excl_keep;
    gc_i64 excl_keep_n, excl_keep_cap;
    gc_i32 *excl_keep_wid;      /* [pitch] each slot's water, or -1      */
    gc_i64 excl_keep_wid_cap;
    gc_i32 excl_keep_valid;     /* derived from this build's records     */

    gc_i64 num_water;
    gc_i32 *water_slot;         /* [3*num_water], component major        */
    gc_i32 *water_group;        /* [num_water], each water's group       */
    gc_i64 num_hgroup;
    gc_i32 *hgr_heavy;          /* [num_hgroup]                          */
    gc_i32 *hgr_group;          /* [num_hgroup], each group's index      */
    gc_i32 *hgr_h;              /* [GC_MAX_HGROUP_H*num_hgroup]          */
    gc_f64 *hgr_dist;           /* [GC_MAX_HGROUP_H*num_hgroup]          */
    gc_f64 *hgr_inv_mass_h;     /* [GC_MAX_HGROUP_H*num_hgroup]          */
    gc_f64 *hgr_inv_mass_c;     /* [num_hgroup]                          */
    gc_i32 *hgr_arity;          /* [num_hgroup]                          */
    gc_i64 water_cap, hgroup_cap;   /* rows the arrays above hold        */
    gc_i32 rigid_ready;         /* counts match this rank's groups; cleared on migration */
    void *rigid_tmp;            /* native_build_rigid's flags and scans  */
    gc_i64 rigid_tmp_cap;       /* bytes                                 */
    int rigid_counted;          /* rigid_tmp holds the current counts, read back */
    /* the rigid descriptor: representative GIDs in ascending order, their
     * H counts and bond distances; rows travel with migrating groups */
    gc_gid *rigid_gid;
    gc_i32 *rigid_arity;
    gc_f64 *rigid_dist;
    gc_i64  rigid_count;
    /* a hydrogen group of more than four hydrogens on any rank: the
     * solvers' wide body is compiled in (gcn_kern_vv1/vv2_hgroup) */
    gc_i32  rigid_wide;

    gc_f64 *lj12, *lj6, *nb14_lj12, *nb14_lj6;
    gc_f64 *table_ene, *table_grad, *table_ecor, *table_decor;
    /* nonbond_precision = MIXED: FP32 copies for the pair term, one block:
     * the LJ pairs {c12, c6} [num_atom_cls^2], then the gradient and the
     * energy interval records, 3 float2 {b0,b1} {b2,d0} {d1,d2} per row
     * (b the row, d the next row minus it), cutoff_int rows each */
    float2 *nb_mixed;
    struct gcn_nbc *nbc;        /* MIXED: the cluster-pair list          */
    gc_f64 *cmap_coef;
    gc_i32 *cmap_resolution;
    /* positional restraints (native_posres_upload): an open-addressing GID
     * table over the whole restrained set, probed per owned slot, so the
     * term follows an atom through every rebuild and migration */
    gc_i64  posres_count;
    gc_i64  posres_mask;        /* table capacity - 1, a power of two - 1 */
    gc_gid *posres_key;         /* [posres_mask+1], 0 = empty            */
    gc_i32 *posres_val;         /* [posres_mask+1], restraint index      */
    gc_f64 *posres_par;         /* [7*posres_count]: k, w_xyz, ref_xyz   */

    unsigned long long *acc_real;  /* [2*GCN_RE_NSLOT] fixed-point sums  */
    unsigned long long *acc_bond;  /* [2*GCN_BE_NSLOT] fixed-point sums  */
    gc_f64 *acc_recip;          /* [GCN_PE_NSLOT]                        */
    gc_f64 *reduce_partial;     /* [16*GCN_MAX_BLOCKS]                   */
    gc_f64 *reduce_out;         /* [16]                                  */
    gc_i64 *verdict;            /* [GCN_VERDICT_W] sticky deferred-check counts, read at the
                                   next synchronising call (verdict_read), any count fatal:
                                   force coordinates [0], halo forward [1] and return [2],
                                   rigid tables of a reordering rebuild [3]; unresolved term
                                   endpoints [4], unplaced mask terms [5], list traps [6] */
    gc_f64 *guard_pin;          /* pinned [2*GC_GUARD_RING]: d_max and largest move per
                                   guard, in slot guard_number % GC_GUARD_RING */
    gc_i64 *guard_done;         /* pinned: guards the device has written (stored last) */
    gc_i64  guard_seq;          /* guards queued since the state was made */
    gc_i64  guard_read;         /* guards gpu_core_list_guard_flush has returned */
    gc_f64 *guard_max;          /* device [2]: the two largest squared distances, left by
                                   VV1's solver passes (guard_ready) */
    gc_i64 *guard_idx;          /* device: [0] guards finalized; [1] steps that kept their
                                   list with d_max at or past the buffer */
    gc_i64  self_epoch;         /* epoch self_energy was reduced at, -1  */
    gc_f64  self_energy;
    gc_i64 *fail_counter;       /* [2] constraint failures and min group */
    gc_i64 *con_fail;           /* [2] solvers' failure count and least failing gid;
                                   sticky until the synchronising read that re-seeds it */
    /* Launches a step entry has taken but not yet queued, so the next entry
     * queues them fused with its own (gpu_step.cu): step_begin's reference save,
     * nvt_half's half velocity, VV1's/VV2's kick and a plain step's force join.
     * vv1_scale: the pending VV1 kick also takes the thermostat's rescale. */
    gc_i32  ref_pending;
    gc_i32  half_pending;
    gc_i32  vv1_pending;
    gc_i32  vv1_scale;
    /* The last VV2's solvers took the next tick's pass (gcn_tick_prep): vel_ref
     * and vel_half hold what step_begin and nvt_half would leave. */
    gc_i32  tick_prep;
    /* ... and ran the tick itself, so the next gpu_core_thermostat has
     * nothing to launch (gcn_tick_prep.t) */
    gc_i32  tick_fold;
    gc_i32  tick_nblk;
    gc_f64 *tick_part;
    gc_i64  tick_part_cap;
    gc_i32  vv2_pending;
    gc_i32  join_pending;
    /* VV1's solver passes wrote this step's force coordinates
     * (gcn_kern_force_coord); cleared by a rebuild or a state push. */
    gc_i32  view_ready;
    /* VV1's solver passes took this step's guard distances (guard_max). */
    gc_i32  guard_ready;
    /* The plain step as a CUDA graph, one executable per kind and mesh slot,
     * captured once per list build; graph_mode: eager, captured or replayed. */
    cudaGraphExec_t graph[GCN_GRAPH_KINDS][GCX_SLOTS];
    gc_i32  graph_pending;      /* plain steps counted, not yet launched */
    gc_i32  graph_pending_kind;
    gc_i32  graph_pending_slot[16];      /* per pending step (GCN_GRAPH_BLOCK) */
    gc_i64  graph_build[GCN_GRAPH_KINDS][GCX_SLOTS];  /* list_builds it was captured at */
    gc_i64  list_builds;        /* gpu_core_rebuild calls                */
    gc_i32  graph_kind;
    gc_i32  graph_slot;
    /* Copy-engine fused exchanges run only when admitted (gpu_step.cu,
     * ce_admission: a fixed topology rule). */
    gc_i32  ce_fused;
    gc_i32  ce_state;           /* 0 undecided, 2 decided                */
    gc_i64  ce_count;           /* graph_begin calls                     */
    gc_i32  ce_graph;           /* plain-step graphs admitted            */
    gc_i32  graph_fused;        /* every rank's passes and moves can fuse */
    gc_i32  graph_stores;       /* ... and all store into peers' slots   */
    gc_i32  graph_mode;         /* GCN_GRAPH_EAGER / _CAPTURE / _REPLAY  */
    gc_f64 *out_pin;            /* pinned [16]: a step reduction's sums */
    struct gcn_thermo *thermo;     /* device                             */
    struct gcn_thermo *thermo_pin; /* pinned staging of the same layout  */
    cudaEvent_t thermo_up;         /* the last draw upload has left thermo_pin */
    struct gcx_wsum *thermo_wsum;  /* sum over ranks in the tick kernel, or null
                                      (the step then sums read-back words) */
    gc_i32 thermo_wsum_tried;
    struct gcn_thermo_args thermo_args;

    struct gcn_pme_mesh *pme_mesh;
    gc_f64 pme_self_fact;       /* -el_fact*alpha/sqrt(pi)               */
    gc_i32 reciprocal_ready;    /* a PME plan exists and owns acc_recip  */
    gc_i32 pad1;

    /* distributed ownership/transport; `dist` is private to gpu_domain.cu */
    struct gcn_layout layout;
    struct gcn_dist_state *dist;
    gc_i64 slot_generation;
    gc_i64 plan_generation;

    gc_step_plan plan;
    gc_table_desc tab;
    gc_constraint_desc con;
    gc_f64 prune_r2;
    gc_f64 cutoff2;
    gc_f64 cubic_lim;           /* CUTOFF_CUBIC: pair evaluated while density*r2 < cubic_lim */
    gc_f64 pairlistdist2;
    gc_f64 box[3];
    gc_i32 cell_count[3];       /* global cell counts                    */
    gc_i64 native_steps;
    gc_i64 scheduled_rebuilds;
    gc_i64 early_rebuilds;
    gc_i64 last_rebuild_step;
    gc_i64 next_scheduled_step;
    gc_i64 graph_segments;
    gc_i32 tables_attached;
    gc_i32 step_ready;
    gc_i32 list_valid;
    gc_i32 force_move_valid;

    /* graph segments, relaunched while addresses and grid are unchanged */

};

#define GCN_MAX_BLOCKS 1024
#define GCN_VERDICT_W 7     /* words of the sticky verdict          */

__global__ void gcn_kern_reduce_finalize(const gc_f64 *__restrict__ part,
                                         int nblk, int ncomp,
                                         gc_f64 *__restrict__ out);

#ifdef __CUDACC__
#include <cfloat>
/* One atom's two guard distances: the squared distance of x from its
 * list-build position ref and from its start-of-step position prev, each
 * under the minimum image. Every rounding is written out so all callers form
 * the same bits; a non-finite distance counts as +infinity. */
static __device__ __forceinline__ void guard_d2(const double x[3],
                                                const double ref[3],
                                                const double prev[3],
                                                const double box[3],
                                                double *d2, double *s2)
{
    double a = 0.0, b = 0.0;
    for (int c = 0; c < 3; ++c) {
        double d = x[c] - ref[c];
        double e = x[c] - prev[c];
        /* within a quarter box the image shift rounds to zero and the fma
           returns d (bar a zero's sign, which squaring drops): the
           division, the costly part, is skipped for almost every atom */
        const double q = 0.25 * box[c];
        if (!(fabs(d) <= q && q > 0.0)) d = __fma_rn(-box[c], round(__ddiv_rn(d, box[c])), d);
        if (!(fabs(e) <= q && q > 0.0)) e = __fma_rn(-box[c], round(__ddiv_rn(e, box[c])), e);
        a = __fma_rn(d, d, a);
        b = __fma_rn(e, e, b);
    }
    if (!(a <= DBL_MAX)) a = INFINITY;
    if (!(b <= DBL_MAX)) b = INFINITY;
    *d2 = a;
    *s2 = b;
}

/* A warp's largest guard distances into slot[0..1] (non-negative doubles
 * order as their bit patterns do).  Every lane of the warp calls it. */
static __device__ __forceinline__ void guard_max_warp(double m, double ms,
                                                      double *slot)
{
    for (int o = 16; o > 0; o >>= 1) {
        m  = fmax(m,  __shfl_xor_sync(0xffffffffu, m,  o));
        ms = fmax(ms, __shfl_xor_sync(0xffffffffu, ms, o));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax((unsigned long long *)&slot[0],
                  (unsigned long long)__double_as_longlong(m));
        atomicMax((unsigned long long *)&slot[1],
                  (unsigned long long)__double_as_longlong(ms));
    }
}

/* One block's sums of ncomp per-thread values into part[c*nblk+block], added
 * by gcn_kern_reduce_finalize in block order. The sum is the halving tree
 * sh[t] += sh[t + half], formed by the first warp alone with two barriers per
 * quantity. blockDim is a power of two from 32 to 256. */
static __device__ __forceinline__ void block_reduce_sum(double *k, int ncomp,
                                                        double *sh,
                                                        double *part,
                                                        int nblk)
{
    const int nw = (int)blockDim.x >> 5;
    for (int c = 0; c < ncomp; ++c) {
        sh[threadIdx.x] = k[c];
        __syncthreads();
        if (threadIdx.x < 32) {
            double v[8];
#pragma unroll
            for (int j = 0; j < 8; ++j)
                v[j] = j < nw ? sh[threadIdx.x + 32 * j] : 0.0;
#pragma unroll
            for (int h = 4; h > 0; h >>= 1)
                if (h < nw)
#pragma unroll
                    for (int j = 0; j < h; ++j) v[j] += v[j + h];
            double s = v[0];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, o);
            if (threadIdx.x == 0) part[c * nblk + blockIdx.x] = s;
        }
        __syncthreads();
    }
}
#endif

/* FP32 bounds lo/hi around a squared FP64 radius and the coordinate range
   they hold for (gpu_rebuild.cu) */
void gcn_sel_bounds(double r2, const double *box, float *lo_out,
                    float *hi_out, double *cmax_out);

/* The join's operands for a pass forming force slot by slot (a plain step's
 * VV2 solvers): force = real + bonded + reciprocal, in that order. fbx is set
 * at one rank. `overflow` is the fixed-point overflow flag; it makes every
 * folded value NaN. */
struct gcn_join {
    void                     *f;        /* gcn_device::force's elements */
    const unsigned long long *frx, *fbx;
    const gc_f64             *fb, *fp;
    const unsigned int       *overflow;
};

namespace gcn {

gc_status dev_calloc(void **p, gc_i64 bytes);
void      dev_release(void **p);
gc_status dev_sync(cudaStream_t s);

/* A velocity or force field's 3*pitch elements as doubles (host[3*pitch]) and
 * back, whatever the field stores; synchronous. */
gc_status native_vf_get(const struct gcn_device *d, const gcn_vf &f,
                        gc_f64 *host);
gc_status native_vf_put(const struct gcn_device *d, const gcn_vf &f,
                        const gc_f64 *host);
/* nonbond_precision = MIXED: velocities and forces become FP32 (vf32),
 * the imported values rounded to nearest.  Once, at attach_tables. */
gc_status native_vf_narrow(struct gcn_device *d);

/* Drain the stream and name the phase that failed (a CUDA fault is sticky). */
gc_status dev_phase(cudaStream_t s, const char *phase);
/* Host work a rebuild stage defers to a later device wait: host_run() runs
 * what is queued and returns the first failure; host_drain() also empties the
 * queue and clears the failure. Per host thread. */
void host_defer(std::function<gc_status()> work);
gc_status host_run();
gc_status host_drain();
/* The step's deferred checks from a copy of d->verdict (and of con_fail,
 * or null): the first failure, reported, or GC_OK. */
gc_status native_verdict_check(gc_context *ctx, const gc_i64 *v,
                               const gc_i64 *fc);

/* As dev_phase without the synchronisation, for per-step launches. */
gc_status dev_launched(const char *phase);
/* Exclusive scans of n values on d's stream, the total (when asked) at
 * *total on the device; and the rebuild's pinned readback words, which a
 * migration inside the rebuild may use (gpu_rebuild.cu). */
gc_status native_scan(struct gcn_device *d, const gc_i32 *in, gc_i64 *out,
                      gc_i64 n, gc_i64 *total);
gc_status native_scan64(struct gcn_device *d, const gc_i64 *in, gc_i64 *out,
                        gc_i64 n, gc_i64 *total);
gc_i64 *native_rebuild_pin(struct gcn_device *d);
/* The rebuild's migration pack, checked against its classification once its
 * readback has landed. */
struct gcn_pack_check {
    const gc_i64 *packed;
    gc_i64 candidate[3];
    gc_i64 groups, owned;
};
gc_status native_pack_verdict(const gc_context *ctx,
                              const struct gcn_pack_check *pc);

/* gpu_rebuild.cu */
gc_status native_state_alloc(gc_context *ctx);
gc_status native_state_free(gc_context *ctx);
gc_status native_upload_import(gc_context *ctx);
gc_status native_force_coordinates_step(gc_context *ctx);
gc_status native_rebuild(gc_context *ctx, gc_i32 early);
/* the guard's two launches: d_max and the step's largest move into the
 * pinned ring; with over > 0, a step whose d_max reaches it counts in
 * guard_idx[1] */
gc_status native_list_guard_launch(gc_context *ctx, gc_f64 over,
                                   cudaStream_t s);
/* The step's lanes and small buffers (gpu_step.cu): fork `n` lanes from
 * stream, join them back, and release what step_words created. */
gc_status native_lanes_fork(struct gcn_device *d, int n);
gc_status native_lanes_join(struct gcn_device *d, int n);
/* step_begin's pending reference save (and nvt_half's pending half
 * velocity) queued on d->stream (gpu_step.cu) */
gc_status native_ref_join(struct gcn_device *d);
void      native_step_release(struct gcn_device *d);

/* gpu_domain.cu */
gc_status native_dist_prepare_outer(gc_context *ctx);
gc_status native_dist_create(gc_context *ctx);
gc_status native_dist_destroy(gc_context *ctx);
void native_dist_summary(const gc_context *ctx);
/* defer: leave the identity maps' failure count on the device, added to
 * native_dist_verdict's, for the caller's next synchronisation. */
gc_status native_dist_refresh_maps(gc_context *ctx, gc_i32 defer);
/* The halo refresh's device failure count (registration, and the deferred
 * maps): nonzero refuses.  Null at one rank. */
gc_i64   *native_dist_verdict(gc_context *ctx);
gc_status native_dist_forward(gc_context *ctx);
gc_status native_dist_reverse(gc_context *ctx);
void native_dist_return_open(gc_context *ctx);
void native_force_bond_fold_owned(struct gcn_device *d);
/* MIXED over several ranks: the halo return reads the ghosts' fixed-point
 * force words into force_bond_fx, so no fold of force_real or force_bond
 * precedes it; later readers take force_bond_fx (the VV2 join) or fold it
 * (native_force_join) */
int native_dist_mixed_words(const struct gcn_device *d);
/* each resident slot's force-view offset, [3*n] (0 without a halo) */
gc_f64   *native_dist_view_move(gc_context *ctx, gc_i64 *n);
/* CUDA-graph support: whether the next forward+reverse may be captured,
 * and the host side of a replayed pair */
int native_dist_graph_ready(const gc_context *ctx);
/* what of native_dist_graph_ready is fixed at set-up, on this rank: every
 * pass can fuse, and every one stores into the peers' slots */
void native_dist_graph_capability(const gc_context *ctx, int *fused,
                                  int *stores);
int native_dist_ce_candidate(const gc_context *ctx);
gc_status native_dist_step_replay(gc_context *ctx);
gc_status native_dist_vote(gc_context *ctx, gc_status local, const char *phase);
/* The largest `local` over the ranks, in two halves: the reduction starts,
 * the caller queues device work, and the wait returns the maximum (1 on a
 * failed reduction).  The request must not move between the two calls. */
struct gcn_dist_max_req {
    long long in, out;
    int active;
    unsigned char req[16];
};
void native_dist_max_start(gc_context *ctx, gc_i64 local,
                           struct gcn_dist_max_req *r);
gc_i64 native_dist_max_wait(struct gcn_dist_max_req *r);
/* The distributed state's one-word device sum over the ranks (gcx_wsum),
 * or null where there is none (no peer route, more than one node).  Every
 * rank launches it in the same order on the rebuild stream. */
struct gcx_wsum *native_dist_wsum(gc_context *ctx);
/* The step's and the rebuild's status check: collective in the checked
 * build, local in the product build (a failure aborts every rank). */
gc_status native_dist_step_vote(gc_context *ctx, gc_status local,
                                const char *phase);

/* Re-register the whole ghost band from the peer's own pack: an approved
 * migration changes which atoms each side must ghost, so the ghost identity
 * set is stale. Runs one coordinate exchange and writes every accepted record
 * into [num_owned, num_owned + received); native_dist_refresh_maps then proves
 * the table. Failures stay on the device (native_dist_verdict). One rank is
 * GC_OK; a halo not admitted, buffers too small or a refusing peer is
 * GC_E_STATE/GC_E_CAPACITY, never a partial registration. */
gc_status native_dist_halo_refresh(gc_context *ctx);

/* gcn_migration_apply.cu */
/* Grow the device slot capacities to hold at least these counts.  Called
 * before any migration mutation; GC_E_NOMEM when the device cannot. */
gc_status native_migration_reserve(gc_context *ctx, gc_i64 owned,
                                   gc_i64 groups, gc_i64 members,
                                   gc_i64 resident);
/* The approved migration's device publication: compact the departing groups
 * out of the CSR, append the arriving groups, install the arriving atom
 * records (read device-to-device) into owned slots
 * [num_owned, num_owned + narrive_atom) and publish the new counts.
 * arrive_atom_cell[i] is the box slot of arriving atom i. */
gc_status native_migration_apply(gc_context *ctx,
                                 const gc_gid *depart_gid, gc_i64 ndepart,
                                 const gcn_group_migration_wire *arrive_group,
                                 gc_i64 narrive_group,
                                 const gcn_atom_migration_wire *d_arrive_atom,
                                 gc_i64 narrive_atom,
                                 const gc_i32 *arrive_atom_cell,
                                 gc_i64 *moved_atoms_out);
/* The replicated topology of one kind (host arrays, every rank the same):
 * n terms ordered by anchor GID (endpoint 0), the CSR by anchor GID
 * (first[g] .. first[g + 1], ngid + 1 entries), and the parameter payload
 * the ids index. */
struct gcn_term_topo_in {
    gc_i64 n = 0;
    const gc_gid *ep = 0;
    const gc_image *epi = 0;
    const gc_i32 *pid = 0, *img = 0, *pbc = 0;
    const gc_i32 *first = 0;
    gc_i64 param_stride = 0, param_count = 0;
    const gc_f64 *payload = 0;
};
/* Upload every kind's topology (in[GC_TERM_NKIND]) once. */
gc_status native_terms_topology(gc_context *ctx, const gcn_term_topo_in *in,
                                gc_gid ngid);
int native_terms_topology_ready(const gc_context *ctx);
void native_terms_topology_reset(gc_context *ctx);
/* Each kind's pool of this rank: the terms anchored on its n sorted owned
 * GIDs (device), from the topology, before the rebuild's term stage. */
gc_status native_terms_derive(gc_context *ctx, const gc_gid *owned, gc_i64 n);
/* The stock real-space zero-bit rows on the device (the same way): one
 * row is two (GID, image) endpoints. */
struct gcn_mask_row {
    gc_gid a;
    gc_image ia;
    gc_gid b;
    gc_image ib;
};
int native_mask_device(const gc_context *ctx, int host_unique);
/* The rows naming one of the nq sorted GIDs q, with their list indices: host
 * (pinned) copies in list order, valid until the next selection. Optionally
 * with the selection: which endpoints of its rows, and which of `extra`
 * further GIDs, are in the sorted device table `table`; one byte each (a
 * row's two), valid as long as the rows. */
struct gcn_mask_member {
    const gc_gid *table = 0;  gc_i64 ntable = 0;   /* device, sorted     */
    const gc_gid *extra = 0;  gc_i64 nextra = 0;   /* host               */
    const gc_u8 *row = 0, *extra_flag = 0;         /* out (pinned)       */
};
/* native_mask_select's first pass, queued ahead of the caller's next
 * synchronisation for a later native_mask_select of the same query
 * without a member (gpu_rebuild.cu). */
gc_status native_mask_select_prefetch(gc_context *ctx, const gc_gid *q,
                                      gc_i64 nq);
gc_status native_mask_select(gc_context *ctx, const gc_gid *q, gc_i64 nq,
                             const gcn_mask_row **rows, const gc_i64 **index,
                             gc_i64 *n, gcn_mask_member *member = 0);
/* Replace the last selection's rows by `rows`, appended. */
gc_status native_mask_replace(gc_context *ctx, const gcn_mask_row *rows,
                              gc_i64 n);
/* Free the transaction's buffers, events and plans kept across rebuilds. */
void native_migration_release(const gc_context *ctx);
/* The transaction driver: reads this rebuild's classification, transports
 * counts and payload (gcx_migration), validates on the receiver, votes the
 * commit, applies on unanimity and derives the terms of the published
 * ownership. Refusal mutates nothing. *did_migrate is 1 exactly when the pair
 * entered the transaction (both ranks agree); *moved_groups / *moved_atoms are
 * this rank's own shipment. */
gc_status migration_txn_native_run(gc_context *ctx, gc_i64 *moved_groups,
                                   gc_i64 *moved_atoms, gc_i32 *did_migrate,
                                   const struct gcn_pack_check *pack = 0);

/* gpu_force.cu */
gc_status native_attach_tables(gc_context *ctx, const gc_table_desc *t,
                               const gc_constraint_desc *c);
/* The real-space pairs whose j cell is owned (GCN_REAL_INTERIOR), whose j
 * cell is a ghost cell (GCN_REAL_BOUNDARY), or all (GCN_REAL_ALL). */
enum { GCN_REAL_ALL = -1, GCN_REAL_INTERIOR = 0, GCN_REAL_BOUNDARY = 1 };
gc_status native_force_real_clear(gc_context *ctx);
gc_status native_force_real(gc_context *ctx, gc_i32 want_energy,
                            cudaStream_t s, int part);
gc_status native_force_bonded(gc_context *ctx, gc_i32 want_energy);
/* keep: force_real and force_bond are read after the join (the virial,
 * r-RESPA's kicks), so one rank folds them rather than summing the
 * fixed-point fields straight into force */
gc_status native_force_join(gc_context *ctx, gc_i32 want_virial,
                            gc_i32 publish, gc_i32 keep,
                            gc_step_result *out);
gc_status native_force_join_view(gc_context *ctx, struct gcn_join *j);

/* gpu_pme.cu */
gc_status native_pme_plan(gc_context *ctx, const gc_pme_desc *d);
gc_status native_pme_release(gc_context *ctx);
/* want_scalars: the step reads the reciprocal energy or virial (the
 * join publishes); otherwise the solve skips its seven sums.  forked: the
 * caller recorded pme_fork on `stream` where the sum forks, so the host may
 * enqueue other work first (a move over the network blocks the host). */
gc_status native_pme_run(gc_context *ctx, gc_i32 want_energy,
                         gc_i32 want_scalars, int forked = 0);
/* whether the mesh spans nodes, so that a move holds the host */
int native_pme_crosses_nodes(const gc_context *ctx);
/* orders `stream` after the reciprocal sum native_pme_run enqueued */
gc_status native_pme_join(gc_context *ctx);
/* CUDA-graph support: whether the next native_pme_run may be captured,
 * and the host side of a replayed one */
int native_pme_graph_ready(const gc_context *ctx);
/* the slot (epoch % GCX_SLOTS) of the next run's mesh moves when one of
 * them is slot-bound (MeshExchangeSession::slot_bound), otherwise 0 */
int native_pme_graph_slot(const gc_context *ctx);
/* what of native_pme_graph_ready is fixed at set-up, on this rank: every
 * move it runs can run fused, and every one stores (else it is fused only
 * with the copy engines admitted) */
void native_pme_graph_capability(const gc_context *ctx, int *fused,
                                 int *stores);
/* whether a mesh move could run fused with copy-engine copies, and
 * whether it may (gpu_step.cu, the copy-engine admission) */
int native_pme_ce_candidate(const gc_context *ctx);
void native_pme_allow_ce(gc_context *ctx, int on);
gc_status native_pme_step_replay(gc_context *ctx);
gc_f64    native_pme_self_energy(gc_context *ctx);
/* the reciprocal parameters of the current box (constant pressure) */
gc_status native_pme_rebox(gc_context *ctx);

/* gpu_step.cu */
gc_status native_reduce(gc_context *ctx, gc_f64 *dst, gc_i32 ncomp);
gc_status native_reduce_n(gc_context *ctx, gc_i64 nitem, gc_f64 *dst,
                          gc_i32 ncomp);
/* The rigid tables of the current groups.  The readbacks of its checks
 * stay pending (native_build_rigid_finish reads them after the caller's
 * next synchronisation, or synchronises itself). */
struct gcn_rigid_pending { int arity, excl; };
gc_status native_build_rigid(gc_context *ctx, struct gcn_rigid_pending *pend);
int native_rigid_recount(const struct gcn_device *d);
gc_status native_rigid_count_early(gc_context *ctx);
gc_status native_build_rigid_finish(gc_context *ctx,
                                    const struct gcn_rigid_pending *pend);

}  /* namespace gcn */

#endif /* GPU_CORE_NATIVE_H */
