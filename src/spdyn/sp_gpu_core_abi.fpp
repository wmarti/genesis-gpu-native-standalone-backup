!--------1---------2---------3---------4---------5---------6---------7---------8
!
!  Module   sp_gpu_core_abi_mod
!> @brief   Fortran mirror of the device-native core's C ABI
!! @authors GENESIS device-native core
!
!  The one-for-one mirror of src/spdyn/gpu_core_abi.h.
!
!--------1---------2---------3---------4---------5---------6---------7---------8

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

module sp_gpu_core_abi_mod

  use, intrinsic :: iso_c_binding

  implicit none
  public

  integer,            parameter :: GcMaxTermArityI = 8
  integer(c_int32_t), parameter :: GcMaxHGroupH    = 8
  integer,            parameter :: GcMaxParamField = 4

  ! status codes
  !
  integer(c_int32_t), parameter :: GcOk           =  0
  integer(c_int32_t), parameter :: GcErrArg       =  1
  integer(c_int32_t), parameter :: GcErrCapacity  =  3
  integer(c_int32_t), parameter :: GcErrUnsupport =  4
  integer(c_int32_t), parameter :: GcErrNoMem     =  6
  integer(c_int32_t), parameter :: GcErrMismatch  =  8
  integer(c_int32_t), parameter :: GcErrEndpoint  = 11
  integer(c_int32_t), parameter :: GcErrArity     = 12
  integer(c_int32_t), parameter :: GcErrState     = 13

  ! term kinds, in the C enumeration's order
  !
  integer(c_int32_t), parameter :: GcTermBond     = 0
  integer(c_int32_t), parameter :: GcTermAngle    = 1
  integer(c_int32_t), parameter :: GcTermDihedral = 2
  integer(c_int32_t), parameter :: GcTermImproper = 3
  integer(c_int32_t), parameter :: GcTermCmap     = 4
  integer(c_int32_t), parameter :: GcTermNb14     = 5
  integer(c_int32_t), parameter :: GcTermExcl     = 6
  integer(c_int32_t), parameter :: GcTermNKind    = 7

  ! energy slots one native force evaluation produces
  !
  integer, parameter :: GcEneBond      =  1
  integer, parameter :: GcEneAngle     =  2
  integer, parameter :: GcEneUrey      =  3
  integer, parameter :: GcEneDihedral  =  4
  integer, parameter :: GcEneImproper  =  5
  integer, parameter :: GcEneCmap      =  6
  integer, parameter :: GcEneElec14    =  7
  integer, parameter :: GcEneVdw14     =  8
  integer, parameter :: GcEneElecReal  =  9
  integer, parameter :: GcEneVdwReal   = 10
  integer, parameter :: GcEneElecRecip = 11
  integer, parameter :: GcEneElecSelf  = 12
  integer, parameter :: GcEneElecCorr  = 13
  integer, parameter :: GcEnePosres    = 14
  integer, parameter :: GcEneNSlot     = 14

  ! the bonded slots as exact words, three doubles each (gc_exact_slot)
  !
  integer, parameter :: GcExactBond    =  1
  integer, parameter :: GcExactAngle   =  2
  integer, parameter :: GcExactUrey    =  3
  integer, parameter :: GcExactDihe    =  4
  integer, parameter :: GcExactImpr    =  5
  integer, parameter :: GcExactCmap    =  6
  integer, parameter :: GcExactElec14  =  7
  integer, parameter :: GcExactVdw14   =  8
  integer, parameter :: GcExactElecCor =  9
  integer, parameter :: GcExactVirX    = 10
  integer, parameter :: GcExactPosres  = 13
  integer, parameter :: GcExactPVirX   = 14
  integer, parameter :: GcExactNSlot   = 16
  integer, parameter :: GcExactNWord   = 3*GcExactNSlot

  ! ensembles, thermostats, kinetic conventions and step modes
  !
  integer(c_int32_t), parameter :: GcEnsembleNVE       = 0
  integer(c_int32_t), parameter :: GcEnsembleNVT       = 1
  integer(c_int32_t), parameter :: GcEnsembleNPT       = 2
  integer(c_int32_t), parameter :: GcThermoNone        = 0
  integer(c_int32_t), parameter :: GcThermoBerend      = 1
  integer(c_int32_t), parameter :: GcThermoBussi       = 2
  integer(c_int32_t), parameter :: GcThermoNHC         = 3
  integer(c_int32_t), parameter :: GcVv1Kick           = 0
  integer(c_int32_t), parameter :: GcVv1FromRef        = 1
  integer(c_int32_t), parameter :: GcVv1DeviceScale    = 2
  integer(c_int32_t), parameter :: GcThermoDraws       = 64
  integer,            parameter :: GcNhcMax            = 10
  integer(c_int32_t), parameter :: GcKinFlatVel        = 0
  integer(c_int32_t), parameter :: GcKinFlatVelRef     = 1
  integer(c_int32_t), parameter :: GcKinFlatVelHalf    = 2
  integer(c_int32_t), parameter :: GcKinGroupVel       = 3
  integer(c_int32_t), parameter :: GcKinGroupVelRef    = 4
  integer(c_int32_t), parameter :: GcKinGroupVelHalf   = 5
  integer(c_int32_t), parameter :: GcConstrainVV1      = 0
  integer(c_int32_t), parameter :: GcConstrainVV2      = 1
  integer(c_int32_t), parameter :: GcConstrainDefer    = 2
  integer(c_int32_t), parameter :: GcConstrainTickNext = 4
  integer(c_int32_t), parameter :: GcGuardRing         = 64
  integer(c_int32_t), parameter :: GcGraphNone         = 0
  integer(c_int32_t), parameter :: GcGraphPlain        = 1
  integer(c_int32_t), parameter :: GcGraphTick         = 2
  integer(c_int32_t), parameter :: GcTablePmeLinear    = 0
  integer(c_int32_t), parameter :: GcTableCutoffCubic  = 1
  integer(c_int32_t), parameter :: GcTablePmeElec      = 2
  integer(c_int32_t), parameter :: GcNonbondDouble     = 0
  integer(c_int32_t), parameter :: GcNonbondMixed      = 1
  integer(c_int32_t), parameter :: GcImproperHarmonic  = 0
  integer(c_int32_t), parameter :: GcImproperFourier   = 1
  integer(c_int32_t), parameter :: GcNb14Table         = 0
  integer(c_int32_t), parameter :: GcNb14Scaled        = 1
  integer(c_int32_t), parameter :: GcEwaldTable        = 0
  integer(c_int32_t), parameter :: GcEwaldAnalytic     = 1
  integer(c_int32_t), parameter :: GcEwaldAuto         = 2

  ! The descriptors.  Every field starts zero (or null), so a caller sets
  ! only what it has.
  !
  type, bind(c) :: s_gc_geometry
    integer(c_int32_t) :: cell(3) = 0
    integer(c_int32_t) :: num_domain(3) = 0
    integer(c_int32_t) :: cell_start(3) = 0
    integer(c_int32_t) :: cell_end(3) = 0
    integer(c_int32_t) :: ncell_local = 0
    integer(c_int32_t) :: ncell_boundary = 0
    real(c_double)     :: system_size(3) = 0.0_c_double
    real(c_double)     :: cell_size(3) = 0.0_c_double
    real(c_double)     :: origin(3) = 0.0_c_double
    real(c_double)     :: pairlistdist = 0.0_c_double
    real(c_double)     :: cutoffdist = 0.0_c_double
    real(c_double)     :: table_support_radius = 0.0_c_double
    real(c_double)     :: prune_support_radius = 0.0_c_double
    integer(c_int32_t) :: nbupdate_period = 0
    integer(c_int32_t) :: host_fp64 = 0
  end type s_gc_geometry

  type, bind(c) :: s_gc_cell
    integer(c_int64_t) :: ncell = 0
    type(c_ptr)        :: cell_l2gx = c_null_ptr
    type(c_ptr)        :: cell_l2gy = c_null_ptr
    type(c_ptr)        :: cell_l2gz = c_null_ptr
    type(c_ptr)        :: cell_l2gx_orig = c_null_ptr
    type(c_ptr)        :: cell_l2gy_orig = c_null_ptr
    type(c_ptr)        :: cell_l2gz_orig = c_null_ptr
    type(c_ptr)        :: cell_tie_key = c_null_ptr
  end type s_gc_cell

  type, bind(c) :: s_gc_atom
    integer(c_int64_t) :: ncell = 0
    type(c_ptr)        :: num_atom = c_null_ptr
    integer(c_int64_t) :: gid_cell_stride = 0
    integer(c_int64_t) :: xyz_cell_stride = 0
    type(c_ptr)        :: gid = c_null_ptr
    type(c_ptr)        :: coord = c_null_ptr
    type(c_ptr)        :: velocity = c_null_ptr
    type(c_ptr)        :: charge = c_null_ptr
    type(c_ptr)        :: mass = c_null_ptr
    type(c_ptr)        :: inv_mass = c_null_ptr
    type(c_ptr)        :: atom_cls = c_null_ptr
  end type s_gc_atom

  type, bind(c) :: s_gc_group
    integer(c_int64_t) :: ncell = 0
    integer(c_int32_t) :: water_atom_count = 0
    type(c_ptr)        :: num_water = c_null_ptr
    integer(c_int64_t) :: water_cell_stride = 0
    type(c_ptr)        :: water_list = c_null_ptr
    integer(c_int32_t) :: hgr_max_h = 0
    type(c_ptr)        :: hgr_local = c_null_ptr
    integer(c_int64_t) :: hgr_local_h_stride = 0
    integer(c_int64_t) :: hgr_local_cell_stride = 0
    type(c_ptr)        :: hgr_bond_list = c_null_ptr
    integer(c_int64_t) :: hgr_list_member_stride = 0
    integer(c_int64_t) :: hgr_list_group_stride = 0
    integer(c_int64_t) :: hgr_list_h_stride = 0
    integer(c_int64_t) :: hgr_list_cell_stride = 0
  end type s_gc_group

  type, bind(c) :: s_gc_term
    integer(c_int32_t) :: kind = 0
    integer(c_int32_t) :: arity = 0
    integer(c_int64_t) :: ncell = 0
    type(c_ptr)        :: num_term = c_null_ptr
    integer(c_int64_t) :: list_cell_stride = 0
    type(c_ptr)        :: list = c_null_ptr
    integer(c_int32_t) :: param_nreal = 0
    integer(c_int32_t) :: param_nint = 0
    type(c_ptr)        :: param_real(GcMaxParamField) = c_null_ptr
    type(c_ptr)        :: param_int(GcMaxParamField) = c_null_ptr
    integer(c_int64_t) :: param_cell_stride = 0
    type(c_ptr)        :: pbc = c_null_ptr
    integer(c_int64_t) :: pbc_cell_stride = 0
    integer(c_int64_t) :: endpoint_count = 0
    type(c_ptr)        :: endpoint_image = c_null_ptr
    type(c_ptr)        :: endpoint_cell = c_null_ptr
    type(c_ptr)        :: endpoint_slot = c_null_ptr
  end type s_gc_term

  !> the 1-4 and excluded pairs, which GENESIS stores per cell pair
  type, bind(c) :: s_gc_pairterm
    integer(c_int32_t) :: kind = 0
    integer(c_int64_t) :: ncell_local = 0
    type(c_ptr)        :: num_term = c_null_ptr
    type(c_ptr)        :: list = c_null_ptr
    integer(c_int64_t) :: list_term_stride = 0
    integer(c_int64_t) :: list_cell_stride = 0
    type(c_ptr)        :: qq_scale = c_null_ptr
    type(c_ptr)        :: lj_scale = c_null_ptr
    integer(c_int64_t) :: scale_cell_stride = 0
  end type s_gc_pairterm

  type, bind(c) :: s_gc_state
    type(c_ptr)        :: geometry = c_null_ptr
    type(c_ptr)        :: cells = c_null_ptr
    type(c_ptr)        :: atoms = c_null_ptr
    type(c_ptr)        :: groups = c_null_ptr
    type(c_ptr)        :: terms = c_null_ptr
    integer(c_int32_t) :: num_terms = 0
    type(c_ptr)        :: pairterms = c_null_ptr
    integer(c_int32_t) :: num_pairterms = 0
    integer(c_int32_t) :: rank = 0
    integer(c_int32_t) :: nproc = 0
    integer(c_int32_t) :: replica = 0
    integer(c_int32_t) :: comm = 0
  end type s_gc_state

  !> the static force-field tables, uploaded once
  type, bind(c) :: s_gc_table
    integer(c_int32_t) :: num_atom_cls = 0
    integer(c_int32_t) :: cutoff_int = 0
    real(c_double)     :: density = 0.0_c_double
    real(c_double)     :: cutoffdist = 0.0_c_double
    type(c_ptr)        :: nonb_lj12 = c_null_ptr
    type(c_ptr)        :: nonb_lj6 = c_null_ptr
    type(c_ptr)        :: nb14_lj12 = c_null_ptr
    type(c_ptr)        :: nb14_lj6 = c_null_ptr
    type(c_ptr)        :: table_ene = c_null_ptr
    type(c_ptr)        :: table_grad = c_null_ptr
    type(c_ptr)        :: table_ecor = c_null_ptr
    type(c_ptr)        :: table_decor = c_null_ptr
    integer(c_int32_t) :: cmap_ntype = 0
    integer(c_int32_t) :: cmap_ngrid = 0
    type(c_ptr)        :: cmap_resolution = c_null_ptr
    type(c_ptr)        :: cmap_coef = c_null_ptr
    integer(c_int32_t) :: table_form = 0
    integer(c_int32_t) :: nonbond_precision = 0
    real(c_double)     :: water_oh_bond = 0.0_c_double
    real(c_double)     :: water_oh_force = 0.0_c_double
    real(c_double)     :: water_hh_bond = 0.0_c_double
    real(c_double)     :: water_hh_force = 0.0_c_double
    real(c_double)     :: water_hoh_angle = 0.0_c_double
    real(c_double)     :: water_hoh_force = 0.0_c_double
    integer(c_int32_t) :: water_bond_calc = 0
    integer(c_int32_t) :: water_bond_hh = 0
    integer(c_int32_t) :: water_angle_calc = 0
    integer(c_int32_t) :: nb14_form = 0
    integer(c_int32_t) :: periodicity_mod = 0
    integer(c_int32_t) :: improper_form = 0
    integer(c_int32_t) :: ewald_evaluation = 0
  end type s_gc_table

  type, bind(c) :: s_gc_constraint
    integer(c_int32_t) :: rigid_bond = 0
    integer(c_int32_t) :: fast_water = 0
    integer(c_int32_t) :: shake_iteration = 0
    real(c_double)     :: shake_tolerance = 0.0_c_double
    real(c_double)     :: water_mass_o = 0.0_c_double
    real(c_double)     :: water_mass_h = 0.0_c_double
    real(c_double)     :: water_r_oh = 0.0_c_double
    real(c_double)     :: water_r_hh = 0.0_c_double
  end type s_gc_constraint

  type, bind(c) :: s_gc_pme
    integer(c_int32_t) :: ngrid(3) = 0
    integer(c_int32_t) :: n_bspline = 0
    real(c_double)     :: alpha = 0.0_c_double
    real(c_double)     :: dielec_const = 0.0_c_double
    real(c_double)     :: elecoef = 0.0_c_double
  end type s_gc_pme

  type, bind(c) :: s_gc_step_plan
    integer(c_int32_t) :: ensemble = 0
    integer(c_int32_t) :: thermostat = 0
    integer(c_int32_t) :: group_tp = 0
    integer(c_int32_t) :: rigid_bond = 0
    real(c_double)     :: dt = 0.0_c_double
    real(c_double)     :: half_dt = 0.0_c_double
    integer(c_int32_t) :: nbupdate_period = 0
    integer(c_int32_t) :: thermo_period = 0
  end type s_gc_step_plan

  type, bind(c) :: s_gc_step_result
    real(c_double)     :: energy(GcEneNSlot) = 0.0_c_double
    real(c_double)     :: virial(3) = 0.0_c_double
    real(c_double)     :: virial_ext(3) = 0.0_c_double
    real(c_double)     :: virial_nb(3) = 0.0_c_double
    real(c_double)     :: bonded_exact(GcExactNWord) = 0.0_c_double
  end type s_gc_step_result

  ! The C entry points (gpu_core_abi.h), all returning a gc_status unless
  ! declared a subroutine.
  !
  interface

    subroutine gcx_place_device(comm, nd, box, link_gbs, units) bind(c)
      import
      integer(c_int), value :: comm
      integer(c_int32_t)    :: nd(3)
      real(c_double)        :: box(3), link_gbs, units
    end subroutine

    integer(c_int32_t) function gcx_route_select(mode) bind(c)
      import
      integer(c_int32_t) :: mode(3)
    end function

    type(c_ptr) function gpu_core_status_string(status) bind(c)
      import
      integer(c_int32_t), value :: status
    end function

    integer(c_int32_t) function gpu_core_try_setup(state, reason) bind(c)
      import
      type(c_ptr), value :: state
      type(c_ptr)        :: reason
    end function

    integer(c_int32_t) function gpu_core_create(state, ctx) bind(c)
      import
      type(c_ptr), value :: state
      type(c_ptr)        :: ctx
    end function

    integer(c_int32_t) function gpu_core_destroy(ctx) bind(c)
      import
      type(c_ptr), value :: ctx
    end function

    integer(c_int32_t) function gpu_core_import(ctx, state) bind(c)
      import
      type(c_ptr), value :: ctx, state
    end function

    integer(c_int32_t) function gpu_core_import_real_mask(ctx, ncell,      &
                         max_atom, num_atom, gid, near_pairs, pairlist,    &
                         mask_self, mask_near) bind(c)
      import
      type(c_ptr),        value :: ctx, num_atom, gid, pairlist
      type(c_ptr),        value :: mask_self, mask_near
      integer(c_int64_t), value :: ncell, max_atom, near_pairs
    end function

    integer(c_int32_t) function gpu_core_counts(ctx, owned, halo) bind(c)
      import
      type(c_ptr), value :: ctx
      integer(c_int64_t) :: owned, halo
    end function

    integer(c_int32_t) function gpu_core_list_guard_plan(ctx,             &
                         pairlistdist, support_radius, half_skin) bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: pairlistdist, support_radius, half_skin
    end function

    integer(c_int32_t) function gpu_core_attach_tables(ctx, tables, con,  &
                         rigid_count, rigid_gid, rigid_arity, rigid_dist, &
                         pme) bind(c)
      import
      type(c_ptr),        value :: ctx, tables, con, pme
      integer(c_int64_t), value :: rigid_count
      type(c_ptr),        value :: rigid_gid, rigid_arity, rigid_dist
    end function

    integer(c_int32_t) function gpu_core_step_setup(ctx, plan, reason)    &
                         bind(c)
      import
      type(c_ptr), value :: ctx, plan
      type(c_ptr)        :: reason
    end function

    integer(c_int32_t) function gpu_core_step_begin(ctx, istep,           &
                         want_energy) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int64_t), value :: istep
      integer(c_int32_t), value :: want_energy
    end function

    integer(c_int32_t) function gpu_core_graph_begin(ctx, kind) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: kind
    end function

    integer(c_int32_t) function gpu_core_graph_end(ctx, next_kind) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: next_kind
    end function

    integer(c_int32_t) function gpu_core_nvt_half(ctx) bind(c)
      import
      type(c_ptr), value :: ctx
    end function

    integer(c_int32_t) function gpu_core_kinetic(ctx, which, kin3, ekin)  &
                         bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: which
      real(c_double)            :: kin3(*), ekin
    end function

    integer(c_int32_t) function gpu_core_kinetic_pair(ctx, which_a,       &
                         which_b, kin_a3, ekin_a, kin_b3, ekin_b) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: which_a, which_b
      real(c_double)            :: kin_a3(*), kin_b3(*), ekin_a, ekin_b
    end function

    integer(c_int32_t) function gpu_core_thermostat_init(ctx, thermostat, &
                         nh_length, nh_step, degree, kboltz, temp0,       &
                         dt_tau, factor, kbt, nh_dt, nh_mass, nh_vel,     &
                         nh_force, nh_coef, kin6) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: thermostat, nh_length, nh_step
      real(c_double),     value :: degree, kboltz, temp0, dt_tau
      real(c_double),     value :: factor, kbt
      real(c_double)            :: nh_dt(*), nh_mass(*), nh_vel(*)
      real(c_double)            :: nh_force(*), nh_coef(*), kin6(*)
    end function

    integer(c_int32_t) function gpu_core_thermostat_draws(ctx, n, draws)  &
                         bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: n
      real(c_double)            :: draws(*)
    end function

    integer(c_int32_t) function gpu_core_thermostat(ctx, which_half,      &
                         which_ref) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: which_half, which_ref
    end function

    integer(c_int32_t) function gpu_core_thermostat_state(ctx, kin_half3, &
                         ekin_half, kin_full3, ekin_full, scale, nh_vel,  &
                         nh_force, nh_coef) bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: kin_half3(*), kin_full3(*)
      real(c_double)     :: ekin_half, ekin_full, scale
      real(c_double)     :: nh_vel(*), nh_force(*), nh_coef(*)
    end function

    subroutine gpu_core_exact_decode(words, nslot, value) bind(c)
      import
      real(c_double)            :: words(*), value(*)
      integer(c_int32_t), value :: nslot
    end subroutine

    integer(c_int32_t) function gpu_core_dynvars_sums(ctx, rmsg,          &
                         ekin_ref) bind(c)
      import
      type(c_ptr), value :: ctx, rmsg, ekin_ref
    end function

    integer(c_int32_t) function gpu_core_vv1(ctx, scale_vel, from_ref)    &
                         bind(c)
      import
      type(c_ptr),        value :: ctx
      real(c_double),     value :: scale_vel
      integer(c_int32_t), value :: from_ref
    end function

    integer(c_int32_t) function gpu_core_vv2(ctx) bind(c)
      import
      type(c_ptr), value :: ctx
    end function

    integer(c_int32_t) function gpu_core_respa_half(ctx) bind(c)
      import
      type(c_ptr), value :: ctx
    end function

    integer(c_int32_t) function gpu_core_respa_vv1(ctx, scale_vel,        &
                         half_dt_long) bind(c)
      import
      type(c_ptr),    value :: ctx
      real(c_double), value :: scale_vel, half_dt_long
    end function

    integer(c_int32_t) function gpu_core_respa_vv2(ctx, half_dt_long)     &
                         bind(c)
      import
      type(c_ptr),    value :: ctx
      real(c_double), value :: half_dt_long
    end function

    integer(c_int32_t) function gpu_core_constrain(ctx, mode, dt, viri3,  &
                         nfail) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: mode
      real(c_double),     value :: dt
      real(c_double)            :: viri3(*)
      integer(c_int64_t)        :: nfail
    end function

    integer(c_int32_t) function gpu_core_npt_vv1(ctx, size_scale,         &
                         vel_scale) bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: size_scale(*), vel_scale(*)
    end function

    integer(c_int32_t) function gpu_core_npt_vv2(ctx, vel_scale, bmoment) &
                         bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: vel_scale(*), bmoment(*)
    end function

    integer(c_int32_t) function gpu_core_npt_group_scale(ctx, scale)      &
                         bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: scale(*)
    end function

    integer(c_int32_t) function gpu_core_npt_respa_vv1(ctx, vel_scale,    &
                         half_dt_long, size_scale) bind(c)
      import
      type(c_ptr),    value :: ctx
      real(c_double)        :: vel_scale(*), size_scale(*)
      real(c_double), value :: half_dt_long
    end function

    integer(c_int32_t) function gpu_core_npt_respa_vv2(ctx, half_dt_long, &
                         bmoment) bind(c)
      import
      type(c_ptr),    value :: ctx
      real(c_double), value :: half_dt_long
      real(c_double)        :: bmoment(*)
    end function

    integer(c_int32_t) function gpu_core_npt_rattle_end(ctx) bind(c)
      import
      type(c_ptr), value :: ctx
    end function

    integer(c_int32_t) function gpu_core_box_scale(ctx, scale, box, recip)&
                         bind(c)
      import
      type(c_ptr),        value :: ctx
      real(c_double)            :: scale(*), box(*)
      integer(c_int32_t), value :: recip
    end function

    integer(c_int32_t) function gpu_core_group_virial(ctx, viri3) bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: viri3(*)
    end function

    integer(c_int32_t) function gpu_core_list_guard_defer(ctx, armed)     &
                         bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: armed
    end function

    integer(c_int32_t) function gpu_core_list_guard_read(ctx, back, vals) &
                         bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: back
      real(c_double)            :: vals(2)
    end function

    integer(c_int32_t) function gpu_core_list_guard_flush(ctx, d_max,     &
                         count, over) bind(c)
      import
      type(c_ptr), value :: ctx
      real(c_double)     :: d_max(*)
      integer(c_int32_t) :: count
      integer(c_int64_t) :: over
    end function

    integer(c_int32_t) function gpu_core_rebuild(ctx, early) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: early
    end function

    integer(c_int32_t) function gpu_core_force(ctx, want_energy,          &
                         want_virial, result) bind(c)
      import
      type(c_ptr),        value :: ctx, result
      integer(c_int32_t), value :: want_energy, want_virial
    end function

    integer(c_int32_t) function gpu_core_force_respa(ctx, want_energy,    &
                         want_virial, outer, result) bind(c)
      import
      type(c_ptr),        value :: ctx, result
      integer(c_int32_t), value :: want_energy, want_virial, outer
    end function

    integer(c_int32_t) function gpu_core_posres_set(ctx, n, gid, ref, par)&
                         bind(c)
      import
      type(c_ptr),        value :: ctx, gid, ref, par
      integer(c_int64_t), value :: n
    end function

    integer(c_int32_t) function gpu_core_com_removal(ctx, do_trans,       &
                         do_rot) bind(c)
      import
      type(c_ptr),        value :: ctx
      integer(c_int32_t), value :: do_trans, do_rot
    end function

    integer(c_int32_t) function gpu_core_pull_state(ctx, count, gid,      &
                         coord, vel) bind(c)
      import
      type(c_ptr),        value :: ctx, gid, coord, vel
      integer(c_int64_t), value :: count
    end function

    integer(c_int32_t) function gpu_core_pull_final_state(ctx, count,     &
                         gid, coord, vel, force, vel_half) bind(c)
      import
      type(c_ptr),        value :: ctx, gid, coord, vel, force, vel_half
      integer(c_int64_t), value :: count
    end function

    integer(c_int32_t) function gpu_core_map_host_slots(count, ncell,     &
                         max_atom, num_atom, id_l2g, gid, cell_out,       &
                         slot_out) bind(c)
      import
      integer(c_int64_t), value :: count, ncell, max_atom
      type(c_ptr),        value :: num_atom, id_l2g, gid, cell_out, slot_out
    end function

    integer(c_int32_t) function gpu_core_find_host_slots(count, ncell,    &
                         max_atom, num_atom, id_l2g, gid, cell_out,       &
                         slot_out) bind(c)
      import
      integer(c_int64_t), value :: count, ncell, max_atom
      type(c_ptr),        value :: num_atom, id_l2g, gid, cell_out, slot_out
    end function

    integer(c_int32_t) function gpu_core_step_summary(ctx, scheduled,     &
                         early, steps, segments) bind(c)
      import
      type(c_ptr), value :: ctx
      integer(c_int64_t) :: scheduled, early, steps, segments
    end function

    integer(c_int32_t) function gpu_core_remd_rollback(ctx) bind(c)
      import
      type(c_ptr), value :: ctx
    end function

    integer(c_int32_t) function gpu_core_remd_rescale(ctx, factor,        &
                         verdict) bind(c)
      import
      type(c_ptr),    value :: ctx
      real(c_double), value :: factor
      integer(c_int32_t)    :: verdict
    end function

  end interface

contains

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Function      gc_status_text
  !> @brief        the C side's own text for a status code
  !! @param[in]    status : a gc_status value
  !! @return       the text, trimmed
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  function gc_status_text(status) result(text)

    ! formal arguments
    integer,                 intent(in) :: status

    ! return value
    character(64)            :: text

    text = gc_c_string(gpu_core_status_string(int(status, c_int32_t)))
    if (text == ' ') text = 'unknown'

    return

  end function gc_status_text

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Function      gc_c_string
  !> @brief        a NUL-terminated C string as Fortran text
  !! @param[in]    p : the C string, or a null pointer
  !! @return       the text, blank for a null pointer
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  function gc_c_string(p) result(text)

    ! formal arguments
    type(c_ptr),             intent(in) :: p

    ! return value
    character(128)           :: text

    ! local variables
    character(kind=c_char), pointer :: chars(:)
    integer                  :: i

    text = ' '
    if (.not. c_associated(p)) return

    call c_f_pointer(p, chars, [len(text)])
    do i = 1, len(text)
      if (chars(i) == c_null_char) exit
      text(i:i) = chars(i)
    end do

    return

  end function gc_c_string

end module sp_gpu_core_abi_mod
