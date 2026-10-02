/*
 * gpu_core_abi.h : the device-native core's C ABI (see doc/21_GPU_Native.rst).
 *
 * POD and C-interoperable: the Fortran bridge (sp_gpu_core_abi.fpp) mirrors
 * these types with ISO_C_BINDING and the CUDA units include them directly.
 * Nothing here reads the environment.
 */

#ifndef GPU_CORE_ABI_H
#define GPU_CORE_ABI_H

/* Feature symbols (USE_GPU, CUDAGPU, GENESIS_GPU_CHECK, ...) reach every unit
 * through config.h; all units sharing gc_context_s must see the same set. */
#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

#include <stddef.h>

#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef __int32 gc_i32;
typedef __int64 gc_i64;
typedef unsigned __int64 gc_u64;
typedef unsigned char gc_u8;
typedef short gc_i16;
#else
#include <stdint.h>
typedef int32_t  gc_i32;
typedef int64_t  gc_i64;
typedef uint64_t gc_u64;
typedef uint8_t  gc_u8;
typedef int16_t  gc_i16;
#endif

typedef double   gc_f64;

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 1. Limits and status ---- */

/* INT32_MAX - 256; the 256 top values are reserved for sentinels. */
#define GC_MAX_LOCAL_INDEX 2147483391

/* Maximum hydrogens in a group (the central atom is not counted). */
#define GC_MAX_HGROUP_H 8

/* The largest arity of any supported term (CMAP). */
#define GC_MAX_TERM_ARITY 8

/* The reserved "absent" global id. */
#define GC_GID_NONE ((gc_i64)0)

typedef gc_i64 gc_gid;      /* global atom / group id, 1-based, 0 = absent  */
typedef gc_i64 gc_term_id;  /* kind in bits 63..56, ordinal in bits 55..0   */
typedef gc_i64 gc_cell_key; /* ((gz*cell_y)+gy)*cell_x+gx, zero based       */
typedef gc_i64 gc_image;    /* three int16 cell shifts packed, 0 = identity */
typedef gc_i64 gc_epoch;    /* monotone per context, 0 = none published     */

typedef enum gc_status {
    GC_OK            =  0,
    GC_E_ARG         =  1,
    GC_E_ABI         =  2,
    GC_E_CAPACITY    =  3,
    GC_E_UNSUPPORTED =  4,
    GC_E_DEVICE      =  5,
    GC_E_NOMEM       =  6,
    GC_E_OVERFLOW    =  7,
    GC_E_MISMATCH    =  8,
    GC_E_EPOCH       =  9,
    GC_E_OWNER       = 10,
    GC_E_ENDPOINT    = 11,
    GC_E_ARITY       = 12,
    GC_E_STATE       = 13
} gc_status;

typedef enum gc_group_kind {
    GC_GROUP_SINGLE = 0,
    GC_GROUP_WATER  = 1,
    GC_GROUP_HGROUP = 2,
    GC_GROUP_NKIND  = 3
} gc_group_kind;

typedef enum gc_term_kind {
    GC_TERM_BOND     = 0,
    GC_TERM_ANGLE    = 1,
    GC_TERM_DIHEDRAL = 2,
    GC_TERM_IMPROPER = 3,
    GC_TERM_CMAP     = 4,
    GC_TERM_NB14     = 5,
    GC_TERM_EXCL     = 6,
    GC_TERM_NKIND    = 7
} gc_term_kind;

/* Arity and the two owner endpoints of each kind, one-based, in GENESIS's
 * listed order.  Indexed by gc_term_kind.  Defined in gpu_core.cpp. */
extern const gc_i32 gc_term_arity[GC_TERM_NKIND];
extern const gc_i32 gc_term_pbc_codes[GC_TERM_NKIND];
extern const gc_i32 gc_term_owner_endpoint[GC_TERM_NKIND][2];

/* Energy slots of one native force evaluation, finer than GENESIS's totals. */
typedef enum gc_energy_slot {
    GC_ENE_BOND       =  0,
    GC_ENE_ANGLE      =  1,
    GC_ENE_UREY       =  2,
    GC_ENE_DIHEDRAL   =  3,
    GC_ENE_IMPROPER   =  4,
    GC_ENE_CMAP       =  5,
    GC_ENE_ELEC14     =  6,
    GC_ENE_VDW14      =  7,
    GC_ENE_ELEC_REAL  =  8,
    GC_ENE_VDW_REAL   =  9,
    GC_ENE_ELEC_RECIP = 10,
    GC_ENE_ELEC_SELF  = 11,
    GC_ENE_ELEC_CORR  = 12,
    GC_ENE_POSRES     = 13,   /* positional restraints (gpu_core_posres_set) */
    GC_ENE_NSLOT      = 14
} gc_energy_slot;

/* The bonded slots as exact fixed-point sums for the cross-rank reduction
 * (gc_step_result.bonded_exact): three doubles per slot (signed high and low
 * 32 bits of the integer word, and the fraction word below 2^32), each an
 * integer, so a double sum over fewer than 2^21 ranks is exact in any order.
 * gpu_core_exact_decode recovers the global value. A slot that cannot be
 * represented carries NaN. */
typedef enum gc_exact_slot {
    GC_EXACT_BOND    =  0,
    GC_EXACT_ANGLE   =  1,
    GC_EXACT_UREY    =  2,
    GC_EXACT_DIHE    =  3,
    GC_EXACT_IMPR    =  4,
    GC_EXACT_CMAP    =  5,
    GC_EXACT_ELEC14  =  6,
    GC_EXACT_VDW14   =  7,
    GC_EXACT_ELECCOR =  8,
    GC_EXACT_VIRX    =  9,   /* sum of d*w: the virial subtracts it */
    GC_EXACT_POSRES  = 12,
    GC_EXACT_PVIRX   = 13,   /* the virial_ext diagonal             */
    GC_EXACT_NSLOT   = 16
} gc_exact_slot;
#define GC_EXACT_NWORD (3 * GC_EXACT_NSLOT)

/* Supported ensembles; others decline at setup (doc/21_GPU_Native.rst). NPT is
 * the MTK barostat with the group temperature convention and rigid bonds. */
typedef enum gc_ensemble {
    GC_ENSEMBLE_NVE = 0,
    GC_ENSEMBLE_NVT = 1,
    GC_ENSEMBLE_NPT = 2
} gc_ensemble;

typedef enum gc_thermostat {
    GC_THERMOSTAT_NONE      = 0,
    GC_THERMOSTAT_BERENDSEN = 1,
    GC_THERMOSTAT_BUSSI     = 2,
    GC_THERMOSTAT_NHC       = 3
} gc_thermostat;

/* Which velocity array a kinetic reduction reads, in which convention. The
 * group convention (compute_kin_group: a free atom as m*v*v, a rigid group in
 * centre-of-mass form) differs bitwise from the flat one. */
typedef enum gc_kinetic {
    GC_KIN_FLAT_VEL        = 0,  /* calc_kinetic over velocity            */
    GC_KIN_FLAT_VEL_REF    = 1,  /* calc_kinetic over velocity_ref        */
    GC_KIN_FLAT_VEL_HALF   = 2,  /* calc_kinetic over velocity_half       */
    GC_KIN_GROUP_VEL       = 3,  /* compute_kin_group over velocity       */
    GC_KIN_GROUP_VEL_REF   = 4,
    GC_KIN_GROUP_VEL_HALF  = 5
} gc_kinetic;

/* The two constraint half steps: VV1 corrects positions and derives the
 * velocity correction from them; VV2 corrects velocity against the
 * already-constrained positions. */
typedef enum gc_constrain_mode {
    GC_CONSTRAIN_VV1 = 0,
    GC_CONSTRAIN_VV2 = 1,
    /* or'ed in: no virial is reduced (viri3 is zero) and the failure count
     * is read at the next force join or undeferred constraint call */
    GC_CONSTRAIN_DEFER = 2,
    /* or'ed into VV2 when the next step is a device thermostat tick and nothing
     * reads or moves the velocities before it: the mixed mode's VV2 solvers may
     * then fold that tick's reference save, half velocity and kinetic partials
     * into their pass */
    GC_CONSTRAIN_TICK_NEXT = 4
} gc_constrain_mode;

/* Ring of per-step d_max values held until gpu_core_list_guard_flush. */
#define GC_GUARD_RING 64

/* table_form: which nonbonded table the descriptor carries, and so which pair
 * arithmetic reads it. PME_LINEAR: 3 values per row, indexed by
 * cutoff2*density/r2. CUTOFF_CUBIC: 6 values per node (cubic Hermite),
 * indexed by density*r2. PME_ELEC: PME with vdw CUTOFF, 1 value per row
 * (electrostatics only), LJ plain r^-12/r^-6. A cutoff run attaches no
 * gc_pme_desc. */
typedef enum gc_table_form {
    GC_TABLE_PME_LINEAR   = 0,
    GC_TABLE_CUTOFF_CUBIC = 1,
    GC_TABLE_PME_ELEC     = 2
} gc_table_form;

/* nonbond_precision ([ENERGY] nonbond_precision): arithmetic of the PME
 * real-space pair term. DOUBLE: FP64 throughout (default). MIXED: FP32 on
 * pair-local coordinates (distance, table interpolation, LJ, pair force,
 * in-warp sums); forces, energies and virial still accumulate in fixed point,
 * so a run stays bitwise reproducible. */
typedef enum gc_nonbond_precision {
    GC_NONBOND_DOUBLE = 0,
    GC_NONBOND_MIXED  = 1
} gc_nonbond_precision;

/* Force-field forms, set once by the bridge (gpu_core_describe_forcefield);
 * the kernels never test the format itself.
 *   improper_form   HARMONIC k(phi-phi0)^2 (CHARMM, GROMACS func 2) or FOURIER
 *                   k(1+cos(n phi-phi0)) (AMBER), the dihedral kernel on the
 *                   improper list.
 *   nb14_form       TABLE: switched LJ from the pair table with the 1-4 LJ
 *                   table, unscaled (CHARMM). SCALED: plain r^-12/r^-6 times
 *                   lj_scale, table electrostatics times qq_scale, and the
 *                   reciprocal sum's 1-4 share removed with weight 1-qq_scale
 *                   (AMBER, GROMACS).
 *   periodicity_mod GENESIS's notation_14types: periodicity plus a 1-4 type
 *                   times it; the rotation count is the remainder. 0: as is. */
typedef enum gc_improper_form {
    GC_IMPROPER_HARMONIC = 0,
    GC_IMPROPER_FOURIER  = 1
} gc_improper_form;

typedef enum gc_nb14_form {
    GC_NB14_TABLE  = 0,
    GC_NB14_SCALED = 1
} gc_nb14_form;
/* ewald_evaluation ([ENERGY] ewald_evaluation), the MIXED pair term:
 *   TABLE     GENESIS's lookup table;
 *   ANALYTIC  Ewald real space and the CHARMM-switched LJ in closed form
 *             (FP32); the run stops if the table is not that form;
 *   AUTO      keyword not given: ANALYTIC when the table is that form, else
 *             TABLE. */
typedef enum gc_ewald_evaluation {
    GC_EWALD_TABLE    = 0,
    GC_EWALD_ANALYTIC = 1,
    GC_EWALD_AUTO     = 2
} gc_ewald_evaluation;

/* ---- 2. Descriptors ----
 * Array fields are borrowed pointers into GENESIS's storage with explicit
 * strides, in ELEMENTS of the pointee type, never bytes. */

/* 2.1  Geometry and the frozen cell-pair rule inputs. */
typedef struct gc_geometry_desc {
    gc_i32 cell[3];          /* global cell counts, boundary%num_cells_*   */
    gc_i32 num_domain[3];    /* domain counts per axis                     */
    /* one-based global cell coordinates (domain%cell_start/_end) */
    gc_i32 cell_start[3];
    gc_i32 cell_end[3];
    gc_i32 ncell_local;
    gc_i32 ncell_boundary;
    gc_f64 system_size[3];
    gc_f64 cell_size[3];
    /* boundary%origin_*: atoms are assigned to cells as sp_domain.fpp does */
    gc_f64 origin[3];
    gc_f64 pairlistdist;
    gc_f64 cutoffdist;
    /* Support radius reached by the initialized force table and by the
     * real-space kernel's zero-force prune, read from the tables; the guard
     * (doc/21_GPU_Native.rst) uses the table radius when initialized. */
    gc_f64 table_support_radius;
    gc_f64 prune_support_radius;
    gc_i32 nbupdate_period;
    gc_i32 host_fp64;        /* 1 when the host build is FP64             */
} gc_geometry_desc;

/* 2.2  Cells.  cell_l2g* are one-based global cell coordinates, length
 * ncell_local + ncell_boundary, local cells first. cell_tie_key is the seam
 * tie-break value (doc/21_GPU_Native.rst): num_atom(cell) * nproc_city +
 * owning rank, as assign_cell_atoms builds it; not an atom count. GENESIS does
 * not retain it, so the bridge reconstructs it. */
typedef struct gc_cell_desc {
    gc_i64        ncell;              /* ncell_local + ncell_boundary      */
    /* One-based EXTENDED coordinates: a wrapped boundary cell is at 0 or
     * cell+1. The owner rule works in this space. */
    const gc_i32 *cell_l2gx;
    const gc_i32 *cell_l2gy;
    const gc_i32 *cell_l2gz;
    /* In-range coordinates of the same cells (domain%cell_l2g*_orig); the
     * image key of a ghost in cell i is (extended - in_range) / cell per axis. */
    const gc_i32 *cell_l2gx_orig;
    const gc_i32 *cell_l2gy_orig;
    const gc_i32 *cell_l2gz_orig;
    const gc_f64 *cell_tie_key;        /* [ncell], real(wp) in GENESIS      */
} gc_cell_desc;

/* 2.3  Atoms, as GENESIS holds them: MaxAtom-padded per-cell slots; the
 * native side stores none of the padding.
 * gid[icel*gid_cell_stride + ix], ix in [0, num_atom[icel]).
 * Coordinate-like arrays are (3, MaxAtom, ncell) in Fortran: element
 * (d, ix, icel) is at icel*xyz_cell_stride + ix*3 + d. */
typedef struct gc_atom_desc {
    gc_i64        ncell;
    const gc_i32 *num_atom;           /* [ncell], atoms in each cell       */
    gc_i64        gid_cell_stride;    /* = MaxAtom                         */
    gc_i64        xyz_cell_stride;    /* = 3 * MaxAtom                     */
    const gc_i32 *gid;                /* domain%id_l2g                     */
    const gc_f64 *coord;              /* domain%coord                      */
    const gc_f64 *velocity;           /* domain%velocity                   */
    const gc_f64 *charge;             /* domain%charge, stride gid_cell_*  */
    const gc_f64 *mass;               /* domain%mass                       */
    const gc_f64 *inv_mass;           /* domain%inv_mass                   */
    const gc_i32 *atom_cls;           /* domain%atom_cls_no                */
} gc_atom_desc;

/* 2.4  Groups, as GENESIS holds them.
 *   Water:  water_list is (nwat_atom, MaxWater, ncell) in Fortran: member m of
 *           water iw in cell icel is at
 *           icel*water_cell_stride + iw*water_atom_count + m.
 *   HGroup: HGr_local(j, icel) counts groups with j hydrogens, 1 <= j <=
 *           GC_MAX_HGROUP_H; HGr_bond_list is
 *           (j+1, HGr_local(j,icel), GC_MAX_HGROUP_H, ncell), heavy atom first.
 * Every atom in neither a water nor a hydrogen group is a singleton group.
 * Both lists hold CELL-LOCAL, ONE-BASED slot indices, not global ids (the term
 * lists of 2.5 hold global ids); the import converts both through
 * domain%id_l2g. */
typedef struct gc_group_desc {
    gc_i64        ncell;
    /* water */
    gc_i32        water_atom_count;   /* 3 (TIP3) or 4 (TIP4)              */
    const gc_i32 *num_water;          /* [ncell]                           */
    gc_i64        water_cell_stride;  /* = water_atom_count * MaxWater     */
    const gc_i32 *water_list;
    /* hydrogen groups */
    gc_i32        hgr_max_h;          /* GENESIS's declared max, <= 8      */
    const gc_i32 *hgr_local;          /* [hgr_max_h * ncell], column major */
    gc_i64        hgr_local_h_stride; /* = 1                               */
    gc_i64        hgr_local_cell_stride;
    const gc_i32 *hgr_bond_list;      /* the (j+1, ig, j, icel) array      */
    gc_i64        hgr_list_member_stride; /* = 1                           */
    gc_i64        hgr_list_group_stride;
    gc_i64        hgr_list_h_stride;
    gc_i64        hgr_list_cell_stride;
} gc_group_desc;

/* Largest number of real or integer parameter fields of one term kind. */
#define GC_MAX_PARAM_FIELD 4

/* 2.5  One term kind, as GENESIS holds it.
 * list[icel*list_cell_stride + it*arity + e] is endpoint e (0 based) of term
 * it in cell icel, as a GLOBAL atom id. Parameters are inline, one array per
 * field shaped (MaxTerm, ncell) sharing param_cell_stride; the native side
 * concatenates a term's fields and deduplicates by exact bit pattern
 * (doc/21_GPU_Native.rst). pbc carries stock's ordered relative codes for
 * bonded force evaluation. */
typedef struct gc_term_desc {
    gc_i32        kind;               /* gc_term_kind                      */
    gc_i32        arity;              /* must equal gc_term_arity[kind]    */
    gc_i64        ncell;
    const gc_i32 *num_term;           /* [ncell]                           */
    gc_i64        list_cell_stride;
    const gc_i32 *list;               /* global atom ids                   */
    gc_i32        param_nreal;        /* 0 .. GC_MAX_PARAM_FIELD           */
    gc_i32        param_nint;
    const gc_f64 *param_real[GC_MAX_PARAM_FIELD];
    const gc_i32 *param_int[GC_MAX_PARAM_FIELD];
    gc_i64        param_cell_stride;  /* shared by every parameter field   */
    const gc_i32 *pbc;                /* stock relative codes, code first */
    gc_i64        pbc_cell_stride;    /* codes per padded source cell      */
    /* GENESIS endpoint copy identity; augments the ordered PBC codes above. */
    gc_i64          endpoint_count;
    const gc_image *endpoint_image;
    const gc_i32   *endpoint_cell;
    const gc_i32   *endpoint_slot;
} gc_term_desc;

/* 2.5b  The 1-4 and excluded pairs, which GENESIS stores per CELL PAIR:
 * enefunc%nb14_calc_list and enefunc%nonb_excl_list are (4, term, owner_cell),
 * the two cells and the two CELL-LOCAL, ONE-BASED slots of the endpoints, the
 * outer index being the cell GENESIS's rule assigned the pair to (a third
 * index space, hence its own descriptor). The import resolves them to GIDs; the
 * periodic image is not imported but derived by the native rebuild.
 * qq_scale and lj_scale are null for CHARMM
 * (own 1-4 LJ table, unscaled charges). */
typedef struct gc_pairterm_desc {
    gc_i32        kind;               /* GC_TERM_NB14 or GC_TERM_EXCL    */
    gc_i64        ncell_local;        /* the owner-cell axis             */
    const gc_i32 *num_term;           /* [ncell_local]                   */
    const gc_i32 *list;               /* (4, term, cell), one based      */
    gc_i64        list_term_stride;   /* = 4                             */
    gc_i64        list_cell_stride;
    const gc_f64 *qq_scale;           /* may be null                     */
    const gc_f64 *lj_scale;           /* may be null                     */
    gc_i64        scale_cell_stride;
} gc_pairterm_desc;

/* 2.6  The whole frozen state handed to one import. */
typedef struct gc_state_desc {
    const gc_geometry_desc *geometry;
    const gc_cell_desc     *cells;
    const gc_atom_desc     *atoms;
    const gc_group_desc    *groups;
    const gc_term_desc     *terms;    /* [num_terms] descriptors           */
    gc_i32                  num_terms;
    /* The cell-pair-indexed kinds (2.5b); null when no pair lists yet. */
    const gc_pairterm_desc *pairterms; /* [num_pairterms] descriptors      */
    gc_i32                  num_pairterms;
    gc_i32                  rank;
    gc_i32                  nproc;
    gc_i32                  replica;
    gc_i32                  comm;     /* Fortran MPI communicator handle   */
} gc_state_desc;

/* 2.7  The static force-field tables, O(num_atom_cls^2 + table rows + CMAP
 * grids): uploaded once at setup and again only when a parameter epoch
 * replaces them. table_ene / table_grad hold three interleaved components per
 * row (lj12, lj6, elec), indexed at 3*L; table_ecor / table_decor the
 * reciprocal-space correction on excluded pairs. */
typedef struct gc_table_desc {
    gc_i32        num_atom_cls;
    gc_i32        cutoff_int;          /* rows of the linear tables       */
    gc_f64        density;
    gc_f64        cutoffdist;
    const gc_f64 *nonb_lj12;           /* [num_atom_cls^2], column major  */
    const gc_f64 *nonb_lj6;
    const gc_f64 *nb14_lj12;           /* the 1-4 LJ pair table           */
    const gc_f64 *nb14_lj6;
    const gc_f64 *table_ene;           /* [3*cutoff_int]                  */
    const gc_f64 *table_grad;          /* [3*cutoff_int]                  */
    const gc_f64 *table_ecor;          /* [cutoff_int]                    */
    const gc_f64 *table_decor;         /* [cutoff_int]                    */
    /* CMAP.  coef is (4,4,ngrid,ngrid,ntype) in Fortran order: element
     * (k,j,g2,g1,t) is at ((t*ngrid + g1)*ngrid + g2)*16 + j*4 + k. */
    gc_i32        cmap_ntype;
    gc_i32        cmap_ngrid;          /* the declared grid dimension     */
    const gc_i32 *cmap_resolution;     /* [cmap_ntype]                    */
    const gc_f64 *cmap_coef;
    gc_i32        table_form;          /* gc_table_form                   */
    gc_i32        nonbond_precision;   /* gc_nonbond_precision            */
    /* Flexible water: when the water is not constrained, GENESIS evaluates
     * the O-H (and H-H) bonds and the H-O-H angle of every water_list
     * molecule from these scalars (enefunc%table%water_*), not from its bond
     * and angle lists. */
    gc_f64        water_oh_bond;
    gc_f64        water_oh_force;
    gc_f64        water_hh_bond;
    gc_f64        water_hh_force;
    gc_f64        water_hoh_angle;     /* radians                         */
    gc_f64        water_hoh_force;
    gc_i32        water_bond_calc;     /* 0/1: the O-H pair               */
    gc_i32        water_bond_hh;       /* 0/1: also the H-H pair          */
    gc_i32        water_angle_calc;    /* 0/1                             */
    gc_i32        nb14_form;           /* gc_nb14_form                    */
    gc_i32        periodicity_mod;     /* notation_14types, 0 = none      */
    gc_i32        improper_form;       /* gc_improper_form                */
    gc_i32        ewald_evaluation;    /* gc_ewald_evaluation             */
} gc_table_desc;

/* 2.8  Rigid-group constraint parameters. */
typedef struct gc_constraint_desc {
    gc_i32        rigid_bond;          /* 0 or 1                          */
    gc_i32        fast_water;          /* SETTLE for the water groups     */
    gc_i32        shake_iteration;
    gc_f64        shake_tolerance;
    gc_f64        water_mass_o;
    gc_f64        water_mass_h;
    gc_f64        water_r_oh;
    gc_f64        water_r_hh;
} gc_constraint_desc;

/* 2.9  The reciprocal sum: mesh, alpha and B-spline order are GENESIS's and
 * are preserved with every normalisation derived from them. */
typedef struct gc_pme_desc {
    gc_i32 ngrid[3];
    gc_i32 n_bspline;
    gc_f64 alpha;
    gc_f64 dielec_const;
    gc_f64 elecoef;                    /* GENESIS's ELECOEF               */
} gc_pme_desc;

/* 2.10  The integrator plan: everything the native step needs that is not
 * particle state. A plan outside the supported span gets GC_E_UNSUPPORTED at
 * gpu_core_step_setup. */
typedef struct gc_step_plan {
    gc_i32 ensemble;                   /* gc_ensemble                     */
    gc_i32 thermostat;                 /* gc_thermostat                   */
    gc_i32 group_tp;                   /* group temperature convention    */
    gc_i32 rigid_bond;
    gc_f64 dt;                         /* AKMA units, timestep/AKMA_PS    */
    gc_f64 half_dt;
    gc_i32 nbupdate_period;            /* the maximum list age            */
    gc_i32 thermo_period;
} gc_step_plan;

/* 2.11  Result of one native force evaluation or step. Energies are the
 * gc_energy_slot slots; virials are diagonal (GENESIS's orthorhombic path). */
typedef struct gc_step_result {
    gc_f64 energy[GC_ENE_NSLOT];
    gc_f64 virial[3];
    gc_f64 virial_ext[3];              /* GENESIS's virial_ext diagonal   */
    gc_f64 virial_nb[3];               /* virial without the bonded slots */
    gc_f64 bonded_exact[GC_EXACT_NWORD];
} gc_step_result;

/* Values of nslot consecutive exact slots (GC_EXACT_*) from their words summed
 * over the ranks: words[3*nslot] in, value[nslot] out. */
void gpu_core_exact_decode(const gc_f64 *words, gc_i32 nslot, gc_f64 *value);

/* ---- 3. Opaque handle and the product entry points ---- */

typedef struct gc_context_s gc_context;

const char *gpu_core_status_string(gc_i32 status);

/* Eligibility of the local rank; the collective accept/decline is the
 * bridge's job. reason_out, when non-null, receives a static string. */
gc_status gpu_core_try_setup(const gc_state_desc *state,
                             const char **reason_out);

gc_status gpu_core_create(const gc_state_desc *state, gc_context **out);
gc_status gpu_core_destroy(gc_context *ctx);

/* Import the frozen state and publish epoch 1. No pointer into the
 * descriptors is kept after return. */
gc_status gpu_core_import(gc_context *ctx, const gc_state_desc *state);

/* Import the stock real-space zero bits, independent of the charge-gated
 * reciprocal correction terms. Cell-pair entries are one-based int16 cell
 * ids; masks are byte arrays in (MaxAtom,MaxAtom,cell) Fortran layout. Call
 * once after state import, before the first native rebuild. Single-rank. */
gc_status gpu_core_import_real_mask(gc_context *ctx,
    gc_i64 ncell_local, gc_i64 max_atom,
    const gc_i32 *num_atom, const gc_i32 *gid_padded,
    gc_i64 near_pairs, const gc_i16 *cell_pairlist1,
    const gc_u8 *mask_self, const gc_u8 *mask_near);

/* Owned and halo atom counts, so the caller can size the export buffers. */
gc_status gpu_core_counts(const gc_context *ctx, gc_i64 *owned_atoms,
                          gc_i64 *halo_atoms);

/* Parameters of the list-validity guard (doc/21_GPU_Native.rst): the list
 * radius, the support radius the tables reach and the resulting half skin. */
gc_status gpu_core_list_guard_plan(const gc_context *ctx, gc_f64 *pairlistdist,
                                   gc_f64 *support_radius, gc_f64 *half_skin);

/* ---- 3.1 The native step (doc/21_GPU_Native.rst) ----
 * The device owns the state; the host owns the decisions. Calls move scalars,
 * not coordinates, forces or term lists, except gpu_core_pull_state and
 * gpu_core_push_state. Required order:
 *   attach_tables -> step_setup -> rebuild
 *   per step: step_begin, [kinetic], vv1, constrain(VV1), list_guard,
 *             [rebuild], force, vv2, constrain(VV2), [com_removal],
 *             [pull_state]
 * A call out of order is GC_E_STATE. */

/* Upload the static tables, constraint parameters, rigid hydrogen groups
 * and PME plan. Legal only between an import and the first step. The rigid
 * groups are keyed by their heavy atom: GENESIS's HGr_bond_dist addressing
 * does not survive the native sort, so the bridge flattens it once, sorted
 * by representative gid (rigid_gid[n], rigid_arity[n] in 1 ..
 * GC_MAX_HGROUP_H, rigid_dist[GC_MAX_HGROUP_H*n]). A null pme means no
 * reciprocal sum. */
gc_status gpu_core_attach_tables(gc_context *ctx,
                                 const gc_table_desc *tables,
                                 const gc_constraint_desc *constraints,
                                 gc_i64 rigid_count, const gc_gid *rigid_gid,
                                 const gc_i32 *rigid_arity,
                                 const gc_f64 *rigid_dist,
                                 const gc_pme_desc *pme);

/* Accept or refuse the integrator plan and build the device work space.
 * reason_out, when non-null, receives a static string on refusal. */
gc_status gpu_core_step_setup(gc_context *ctx, const gc_step_plan *plan,
                              const char **reason_out);

/* Open a step: save the reference coordinate and velocity (copy_coord_vel)
 * and, when want_energy, prepare the half-step velocity nve_vv1 derives. */
gc_status gpu_core_step_begin(gc_context *ctx, gc_i64 istep,
                              gc_i32 want_energy);

/* A plain step (no output, energy, virial, host read, list build, COM removal
 * or thermostat draw upload between step_begin and the VV2 constraint) of the
 * given kind, called before step_begin. Its launches are captured in a CUDA
 * graph once per list build and replayed on later plain steps of that kind;
 * gpu_core_graph_end is told the next step's kind so a run of plain steps is
 * launched as one block. GC_GRAPH_NONE or one rank runs the
 * step uncaptured. */
typedef enum gc_graph_kind {
    GC_GRAPH_NONE = 0,
    GC_GRAPH_PLAIN = 1,     /* no thermostat tick */
    GC_GRAPH_TICK = 2       /* a thermostat tick */
} gc_graph_kind;
gc_status gpu_core_graph_begin(gc_context *ctx, gc_i32 kind);
gc_status gpu_core_graph_end(gc_context *ctx, gc_i32 next_kind);

/* On a generic NVT thermostat tick, turn the previous VV2 half velocity into
 * velocity_half - current velocity before the kinetic reduction. */
gc_status gpu_core_nvt_half(gc_context *ctx);

/* One kinetic reduction in the convention gc_kinetic names. kin3 is the
 * per-axis sum of m*v*v and ekin half its total (calc_kinetic,
 * compute_kin_group). Device reduction; the caller does the communicator one. */
gc_status gpu_core_kinetic(const gc_context *ctx, gc_i32 which,
                           gc_f64 *kin3, gc_f64 *ekin);
/* Two tensors of one kind (both flat or both group) in one reduction. */
gc_status gpu_core_kinetic_pair(const gc_context *ctx, gc_i32 which_a,
                                gc_i32 which_b, gc_f64 *kin_a3,
                                gc_f64 *ekin_a, gc_f64 *kin_b3,
                                gc_f64 *ekin_b);
/* Local compute_dynvars sums at the post-VV1 energy-output phase; the shared
 * finalization does the communicator reduce. */
gc_status gpu_core_dynvars_sums(const gc_context *ctx, gc_f64 *rmsg,
                                gc_f64 *ekin_ref);

/* The device thermostat. A tick (gpu_core_thermostat) reduces the half and
 * reference kinetic tensors, sums them over ranks as fixed-point words and
 * forms the Bussi/Berendsen/NHC scale on the device, where
 * gpu_core_vv1(GC_VV1_DEVICE_SCALE) applies it. _init takes the constants in
 * GENESIS's own expressions (nh_dt: dt_1, dt_2, dt_4, dt_8 per weight), the
 * chain and the kin6 tensors published by non-tick steps until the first tick;
 * _draws uploads the (rr, sum_gauss(degree-1)) pairs of the next n <= 64 Bussi
 * ticks in GENESIS's order; _state returns the last tick's state and
 * synchronises. */
gc_status gpu_core_thermostat_init(gc_context *ctx, gc_i32 thermostat,
                                   gc_i32 nh_length, gc_i32 nh_step,
                                   gc_f64 degree, gc_f64 kboltz,
                                   gc_f64 temp0, gc_f64 dt_tau,
                                   gc_f64 factor, gc_f64 kbt,
                                   const gc_f64 *nh_dt, const gc_f64 *nh_mass,
                                   const gc_f64 *nh_vel,
                                   const gc_f64 *nh_force,
                                   const gc_f64 *nh_coef,
                                   const gc_f64 *kin6);
gc_status gpu_core_thermostat_draws(gc_context *ctx, gc_i32 n,
                                    const gc_f64 *draws);
gc_status gpu_core_thermostat(gc_context *ctx, gc_i32 which_half,
                              gc_i32 which_ref);
gc_status gpu_core_thermostat_state(gc_context *ctx, gc_f64 *kin_half3,
                                    gc_f64 *ekin_half, gc_f64 *kin_full3,
                                    gc_f64 *ekin_full, gc_f64 *scale,
                                    gc_f64 *nh_vel, gc_f64 *nh_force,
                                    gc_f64 *nh_coef);

/* The VV1 sweep. from_ref selects the form:
 *   0  v += (half_dt/m) f ; x_ref = x ; x += dt v  (optionally rescaled first,
 *      in the group or flat convention)
 *   1  v = scale*v_ref + (half_dt/m) f ; x = x_ref + dt v  (iterated
 *      constrained thermostat form; reads references it does not write)
 *   2  as 0, rescaled by the scale gpu_core_thermostat left on the device
 *      (scale_vel is ignored) */
enum { GC_VV1_KICK = 0, GC_VV1_FROM_REF = 1, GC_VV1_DEVICE_SCALE = 2 };
gc_status gpu_core_vv1(gc_context *ctx, gc_f64 scale_vel, gc_i32 from_ref);

/* The VV2 half kick; saves velocity_half and, under rigid_bond,
 * velocity_full before the RATTLE that follows. */
gc_status gpu_core_vv2(gc_context *ctx);

/* r-RESPA (integrator = VRES): short force = real space plus bonded, long
 * force = reciprocal sum.
 *   respa_half: velocity_half <- (half_dt/m) f for the thermostat's kinetic term.
 *   respa_vv1: optional rescale, v += (half_dt_long/m) f_long when
 *     half_dt_long > 0, v += (half_dt/m) f_short, x_ref = x, x += dt v.
 *   respa_vv2: the same kicks, saving velocity_full under rigid_bond. */
gc_status gpu_core_respa_half(gc_context *ctx);
gc_status gpu_core_respa_vv1(gc_context *ctx, gc_f64 scale_vel,
                             gc_f64 half_dt_long);
gc_status gpu_core_respa_vv2(gc_context *ctx, gc_f64 half_dt_long);

/* SETTLE and SHAKE at VV1, water and H-group RATTLE at VV2. viri3 receives the
 * diagonal constraint virial (GENESIS's convention), nfail the number of
 * groups that did not converge. */
gc_status gpu_core_constrain(gc_context *ctx, gc_i32 mode, gc_f64 dt,
                             gc_f64 *viri3, gc_i64 *nfail);

/* compute_virial_group: rigid-group virial from the reference coordinates and
 * the current force, over water and hydrogen groups only. */
gc_status gpu_core_group_virial(const gc_context *ctx, gc_f64 *viri3);

/* List-validity guard (doc/21_GPU_Native.rst). defer queues this step's d_max
 * (and largest one-step move) on the device; armed = 1 on a step that keeps its
 * list under [DYNAMICS] gpu_list_guard. flush returns the queued d_max values
 * (at most GC_GUARD_RING, oldest first) for the caller's one MPI_MAX, and in
 * *over this rank's count of armed steps whose d_max reached the list buffer
 * read waits for one queued guard's two values
 * without synchronising the device. */
gc_status gpu_core_list_guard_defer(gc_context *ctx, gc_i32 armed);
gc_status gpu_core_list_guard_flush(gc_context *ctx, gc_f64 *d_max,
                                    gc_i32 *count, gc_i64 *over);
gc_status gpu_core_list_guard_read(gc_context *ctx, gc_i32 back,
                                   gc_f64 *vals);

/* The rebuild transaction: group sort, cell runs, pair list, selected-index
 * runs, masks and execution term records. early selects which of the two
 * reported counters it adds to; an early rebuild does not cancel the next
 * scheduled one. */
gc_status gpu_core_rebuild(gc_context *ctx, gc_i32 early);

/* Positional restraints (compute_energy_restraints_pos):
 * E = k * sum_d w_d (x_d - ref_d)^2 per restrained atom; force on the atom,
 * virial into virial_ext. Each rank passes the restraints of its own cells;
 * the call gathers them over the communicator so an atom keeps its restraint
 * wherever it migrates. par is 4 per restraint (k, w_x, w_y, w_z), ref 3.
 * Collective; call once after gpu_core_import. n = 0 on every rank leaves the
 * term off. */
gc_status gpu_core_posres_set(gc_context *ctx, gc_i64 n, const gc_i32 *gid,
                              const gc_f64 *ref, const gc_f64 *par);

/* One force evaluation over the current epoch: real space, bonded, 1-4,
 * excluded-pair correction and reciprocal sum, joined in a defined order. */
gc_status gpu_core_force(gc_context *ctx, gc_i32 want_energy,
                         gc_i32 want_virial, gc_step_result *out);
/* The same for r-RESPA, whose short (real space plus bonded) and long
 * (reciprocal) forces stay apart. An inner step (outer = 0) has no reciprocal
 * sum: its force, energy and virial stay the last outer step's. */
gc_status gpu_core_force_respa(gc_context *ctx, gc_i32 want_energy,
                               gc_i32 want_virial, gc_i32 outer,
                               gc_step_result *out);

/* The MTK barostat integrator (group temperature convention; scalars are the
 * host's).
 *   npt_vv1: compute_vv1_group from x_ref: a rigid group's centre of mass is
 *     scaled by size_scale and its COM velocity by vel_scale (a free atom: its
 *     coordinate and velocity); then v += (half_dt/m) f ; x += dt v.
 *   npt_vv2: COM velocity and mean force removed over every rank;
 *     velocity_half <- v; v += (half_dt/m) f; COM velocity scaled by vel_scale;
 *     v <- v + bmoment*x, which the velocity RATTLE constrains in the scaled
 *     frame; x_ref and velocity_full keep both halves.
 *   npt_rattle_end: after constrain(VV2), v = velocity_full + (v - x_ref).
 *   box_scale: the box becomes box[3]; every image offset of the force view
 *     scales by scale[3] as stock scales trans_vec; with recip the reciprocal
 *     sum takes the new box. */
gc_status gpu_core_npt_vv1(gc_context *ctx, const gc_f64 *size_scale,
                           const gc_f64 *vel_scale);
gc_status gpu_core_npt_vv2(gc_context *ctx, const gc_f64 *vel_scale,
                           const gc_f64 *bmoment);
gc_status gpu_core_npt_rattle_end(gc_context *ctx);
/* The r-RESPA MTK barostat.
 *   npt_group_scale: update_vel_group_3d: a rigid group's COM velocity (a free
 *     atom: its velocity) times scale per axis.
 *   npt_respa_vv1: on an outer step (half_dt_long > 0) the group scale
 *     vel_scale and the long kick, then the short kick and
 *     compute_vv1_coord_group with size_scale.
 *   npt_respa_vv2: on an outer step the COM velocity and mean long-force
 *     removal and the long kick, then the short kick; under rigid bonds the
 *     RATTLE constrains v + bmoment*x as in npt_vv2, closed by npt_rattle_end. */
gc_status gpu_core_npt_group_scale(gc_context *ctx, const gc_f64 *scale);
gc_status gpu_core_npt_respa_vv1(gc_context *ctx, const gc_f64 *vel_scale,
                                 gc_f64 half_dt_long, const gc_f64 *size_scale);
gc_status gpu_core_npt_respa_vv2(gc_context *ctx, gc_f64 half_dt_long,
                                 const gc_f64 *bmoment);
gc_status gpu_core_box_scale(gc_context *ctx, const gc_f64 *scale,
                             const gc_f64 *box, gc_i32 recip);

/* stop_trans_rotation over the owned atoms. */
gc_status gpu_core_com_removal(gc_context *ctx, gc_i32 do_trans,
                               gc_i32 do_rot);

/* The two declared boundary exceptions. Arrays use the current compact device
 * slot order, reported by gid; count is the owned atom count. */
gc_status gpu_core_pull_state(const gc_context *ctx, gc_i64 count,
                              gc_gid *gid, gc_f64 *coord, gc_f64 *vel);
/* Terminal-only completed-VV2 state for stock's extra final VV1. */
gc_status gpu_core_pull_final_state(const gc_context *ctx, gc_i64 count,
                                    gc_gid *gid, gc_f64 *coord, gc_f64 *vel,
                                    gc_f64 *force, gc_f64 *vel_half);
/* Build a bijection from every occupied stock owned slot (water included) to
 * the current compact device order; output cells and slots are zero based.
 * Called only at a declared boundary. */
gc_status gpu_core_map_host_slots(gc_i64 count, gc_i64 ncell,
                                  gc_i64 max_atom, const gc_i32 *num_atom,
                                  const gc_i32 *id_l2g, const gc_gid *gid,
                                  gc_i32 *cell_out, gc_i32 *slot_out);
/* As above where the host domain may not hold every device atom (more than one
 * rank): an atom it does not hold gets cell and slot -1. */
gc_status gpu_core_find_host_slots(gc_i64 count, gc_i64 ncell,
                                   gc_i64 max_atom, const gc_i32 *num_atom,
                                   const gc_i32 *id_l2g, const gc_gid *gid,
                                   gc_i32 *cell_out, gc_i32 *slot_out);

/* Engagement record (doc/21_GPU_Native.rst): steps run natively and the two
 * rebuild counters. */
gc_status gpu_core_step_summary(const gc_context *ctx, gc_i64 *scheduled,
                                gc_i64 *early, gc_i64 *native_steps,
                                gc_i64 *graph_segments);

/* ---- 4. Checked-only entry points ----
 * Present only when GENESIS_GPU_CHECK is defined; typed API calls made by the
 * checked driver, with no environment variable or spdyn keyword behind them. */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* GPU_CORE_ABI_H */
