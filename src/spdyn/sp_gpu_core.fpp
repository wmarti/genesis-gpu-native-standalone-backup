!--------1---------2---------3---------4---------5---------6---------7---------8
!
!  Module   sp_gpu_core_mod
!> @brief   the only scientific Fortran bridge to the device-native core
!! @authors GENESIS device-native core
!
!  Converts s_domain, s_enefunc, s_constraints and s_boundary into the POD
!  descriptors of gpu_core_abi.h, with explicit strides taken from the arrays
!  themselves. The context handle is opaque, per rank and replica.
!
!--------1---------2---------3---------4---------5---------6---------7---------8

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

module sp_gpu_core_mod

  use sp_gpu_core_abi_mod
  use sp_energy_pme_mod, only: pme_ngrid_used
  use sp_domain_str_mod
  use sp_enefunc_str_mod
  use sp_energy_str_mod, only: VDWCutoff
  use sp_constraints_str_mod
  use sp_boundary_str_mod
  use messages_mod
  use constants_mod
  use mpi_parallel_mod
  use, intrinsic :: iso_c_binding
#ifdef HAVE_MPI_GENESIS
  use mpi
#endif

  implicit none
  private

  !> endpoint selection borrowed from stock id_g2l, one record per term kind
  type :: s_gc_endpoint_capture
    integer(c_int64_t), allocatable :: image(:)
    integer(c_int32_t), allocatable :: cell(:), slot(:)
  end type s_gc_endpoint_capture

  !> @brief one native context and the descriptors that describe its input
  type, public :: s_gpu_core
    type(c_ptr)                  :: ctx = c_null_ptr
    logical                      :: active = .false.
    character(128)               :: reason = 'not attempted'
    type(s_gc_geometry)          :: geometry
    type(s_gc_cell)              :: cells
    type(s_gc_atom)              :: atoms
    type(s_gc_group)             :: groups
    type(s_gc_term), allocatable :: terms(:)
    type(s_gc_endpoint_capture), allocatable :: endpoint_capture(:)
    type(s_gc_pairterm)          :: pairterms(2)
    type(s_gc_state)             :: state
    type(s_gc_table)             :: tables
    type(s_gc_constraint)        :: cons
    type(s_gc_pme)               :: pme
    type(s_gc_step_plan)         :: plan
    !> the list guard's largest displacement since a build, and the steps
    !> past the half skin (gpu_list_guard: kept steps past the whole buffer)
    real(c_double)               :: guard_dmax = 0.0_c_double
    integer(c_int64_t)           :: guard_over = 0_c_int64_t
    logical                      :: list_guard = .false.
    !> the rigid hydrogen groups, flattened and sorted by representative
    integer(c_int64_t), allocatable :: rigid_gid(:)
    integer(c_int32_t), allocatable :: rigid_arity(:)
    real(c_double),     allocatable :: rigid_dist(:)
    !> the owned-atom staging of the output boundaries
    integer(c_int64_t), allocatable :: out_gid(:)
    integer(c_int32_t), allocatable :: out_cell(:), out_slot(:)
    real(c_double),     allocatable :: out_coord(:), out_vel(:)
    real(c_double),     allocatable :: out_force(:), out_vel_half(:)
    integer(c_int64_t)           :: num_owned = 0_c_int64_t
    logical                      :: step_ready = .false.
    !> the list guard's plan (gpu_core_list_guard_plan) and the halo count
    real(c_double)               :: list_radius = 0.0_c_double
    real(c_double)               :: support_radius = 0.0_c_double
    real(c_double)               :: half_skin = 0.0_c_double
    integer(c_int64_t)           :: num_halo = 0_c_int64_t
    !> the seam tie key, rebuilt because GENESIS does not retain it
    real(wp),        allocatable :: tie_key(:)
  end type s_gpu_core

  public  :: gpu_core_vote
  public  :: gpu_core_setup
  public  :: gpu_core_admit_real_mask
  public  :: gpu_core_attach_posres
  public  :: gpu_core_finalize
  public  :: gpu_core_attach_step
  public  :: gpu_core_pull
  public  :: gpu_core_summary

contains

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_vote
  !> @brief        one collective accept-or-decline per phase: the largest
  !!               status over mpi_comm_country, so no rank goes on alone
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_vote(status, nproc)

    integer,                 intent(inout) :: status
    integer,                 intent(in)    :: nproc

#ifdef HAVE_MPI_GENESIS
    integer                  :: global_status, ierror

    if (nproc > 1) then
      call mpi_allreduce(status, global_status, 1, mpi_integer, mpi_max, &
                         mpi_comm_country, ierror)
      status = global_status
      if (ierror /= 0) status = GcErrState
    end if
#else
    if (nproc > 1) status = GcErrUnsupport
#endif

  end subroutine gpu_core_vote

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_build_tie_key
  !> @brief        rebuild the seam tie key assign_cell_atoms produced:
  !!               num_atom(cell)*nproc_city + <owning rank>, exact only
  !!               while num_atom holds the setup-time occupancy
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_build_tie_key(core, domain)

    type(s_gpu_core),        intent(inout) :: core
    type(s_domain),          intent(in)    :: domain

    integer                  :: ncell, i, d(3), lo(3), hi(3), g(3)

    ncell = domain%num_cell_local + domain%num_cell_boundary
    if (allocated(core%tie_key)) deallocate(core%tie_key)
    allocate(core%tie_key(ncell))
    core%tie_key(1:ncell) = 0.0_wp

    if (allocated(domain%cell_tie_key)) then
      if (size(domain%cell_tie_key) >= ncell) then
        core%tie_key(1:ncell) = domain%cell_tie_key(1:ncell)
        return
      end if
    end if

    do i = 1, domain%num_cell_local
      core%tie_key(i) = real(domain%num_atom(i)*nproc_city + my_city_rank, wp)
    end do

    lo(1:3) = domain%cell_start(1:3) - 1
    hi(1:3) = domain%cell_end(1:3)   + 1
    do i = domain%num_cell_local + 1, ncell
      g = [domain%cell_l2gx(i), domain%cell_l2gy(i), domain%cell_l2gz(i)]
      d = merge(-1, 0, g == lo) + merge(1, 0, g == hi)
      core%tie_key(i) = real(domain%num_atom(i)*nproc_city &
                           + domain%neighbor(d(1),d(2),d(3)), wp)
    end do

  end subroutine gpu_core_build_tie_key

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Function      gpu_core_support_radius
  !> @brief        the radius the initialized force table actually reaches,
  !!               which the list-validity guard must read (zero if unknown)
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  function gpu_core_support_radius(enefunc) result(radius)

    type(s_enefunc),         intent(in) :: enefunc
    real(wp)                 :: radius

    integer                  :: i, l1
    real(wp)                 :: cutoff2, density

    radius = 0.0_wp

    if (.not. allocated(enefunc%table%table_grad)) return
    density = enefunc%table%density
    if (density <= 0.0_wp) return
    cutoff2 = enefunc%cutoffdist * enefunc%cutoffdist

    if (.not. enefunc%pme_use) then
      ! the cubic cutoff table is indexed by density*r2 and a pair reads
      ! nodes L and L+1 of L = int(density*r2) (sp_energy_table_cubic.fpp):
      ! with nz the last non-zero node the support is density*r2 < nz+1
      l1 = 0
      do i = 1, enefunc%table%cutoff_int
        if (any(enefunc%table%table_ene (6*i-5:6*i) /= 0.0_wp) .or. &
            any(enefunc%table%table_grad(6*i-5:6*i) /= 0.0_wp)) l1 = i
      end do
      if (l1 > 0) radius = sqrt(real(l1 + 1, wp) / density)
      return
    end if

    ! vdw = CUTOFF under PME: the pair kernel stops at the cutoff itself
    if (enefunc%vdw == VDWCutoff) then
      radius = enefunc%cutoffdist
      return
    end if

    ! The table is indexed by L = int(cutoff2*density/r2) and interpolates
    ! rows L and L+1, so the support is one row farther out than the first
    ! non-zero row's own radius.
    do i = 1, size(enefunc%table%table_grad) / 3
      if (any(enefunc%table%table_grad(3*i-2:3*i) /= 0.0_wp)) then
        radius = sqrt(cutoff2 * density / real(max(i - 1, 1), wp))
        return
      end if
    end do

  end function gpu_core_support_radius

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_add_term
  !> @brief        append one canonical term-kind descriptor to the state
  !
  !  A kind this rank holds no record of (every per-cell count zero; GENESIS
  !  sizes an unused kind's list to zero) has no list to describe. A kind
  !  with records keeps every shape and capacity check of the capture.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_add_term(core, kind, arity, num_term, list, pstride, &
                               pr1, pr2, pr3, pr4, pi1, pbc1, pbcn)

    type(s_gpu_core),        intent(inout) :: core
    integer,                 intent(in)    :: kind, arity
    integer,         target, intent(in)    :: num_term(:)
    integer,         target, intent(in)    :: list(:,:,:)
    integer,                 intent(in)    :: pstride
    real(wp),        target, intent(in), optional :: pr1(:,:), pr2(:,:)
    real(wp),        target, intent(in), optional :: pr3(:,:), pr4(:,:)
    integer,         target, intent(in), optional :: pi1(:,:), pbc1(:,:)
    integer,         target, intent(in), optional :: pbcn(:,:,:)

    type(s_gc_term), allocatable :: grown(:)
    type(s_gc_term)          :: t
    integer                  :: n

    if (size(num_term) == 0) return
    if (all(num_term == 0)) return

    t%kind             = int(kind,  c_int32_t)
    t%arity            = int(arity, c_int32_t)
    t%ncell            = int(size(num_term), c_int64_t)
    t%num_term         = c_loc(num_term(1))
    t%list_cell_stride = int(size(list,1)*size(list,2), c_int64_t)
    t%list             = c_loc(list(1,1,1))
    t%param_cell_stride = int(pstride, c_int64_t)
    if (present(pr1)) t%param_real(1) = c_loc(pr1(1,1))
    if (present(pr2)) t%param_real(2) = c_loc(pr2(1,1))
    if (present(pr3)) t%param_real(3) = c_loc(pr3(1,1))
    if (present(pr4)) t%param_real(4) = c_loc(pr4(1,1))
    t%param_nreal = int(count([present(pr1), present(pr2), present(pr3), &
                               present(pr4)]), c_int32_t)
    if (present(pi1)) then
      t%param_nint   = 1_c_int32_t
      t%param_int(1) = c_loc(pi1(1,1))
    end if
    if (present(pbc1)) then
      t%pbc = c_loc(pbc1(1,1))
      t%pbc_cell_stride = int(size(pbc1,1), c_int64_t)
    end if
    if (present(pbcn)) then
      t%pbc = c_loc(pbcn(1,1,1))
      t%pbc_cell_stride = int(size(pbcn,1)*size(pbcn,2), c_int64_t)
    end if

    n = size(core%terms)
    allocate(grown(n+1))
    grown(1:n) = core%terms(1:n)
    grown(n+1) = t
    call move_alloc(grown, core%terms)

  end subroutine gpu_core_add_term

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_capture_endpoints
  !> @brief        each term endpoint's source cell, slot and packed image
  !!               shift, taken from stock id_g2l and the cell images
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_capture_endpoints(core, domain, status)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),          intent(in)    :: domain
    integer,                 intent(out)   :: status

    integer(c_int32_t), pointer :: count(:), list(:,:,:)
    integer                  :: k, c, t, e, gid, sc, ss, axis
    integer                  :: ncell, nsrc, arity, maxterm
    integer                  :: ext(3), orig(3), delta, width, shift, ierr
    integer(c_int64_t)       :: total, at, packed, limit

    status = GcOk
    if (allocated(core%endpoint_capture)) deallocate(core%endpoint_capture)
    if (size(core%terms) == 0) return
    if (.not. (allocated(domain%id_g2l) .and. allocated(domain%id_l2g) .and.&
               allocated(domain%num_atom) .and.                           &
               allocated(domain%cell_l2gx) .and.                          &
               allocated(domain%cell_l2gy) .and.                          &
               allocated(domain%cell_l2gz) .and.                          &
               allocated(domain%cell_l2gx_orig) .and.                     &
               allocated(domain%cell_l2gy_orig) .and.                     &
               allocated(domain%cell_l2gz_orig))) then
      status = GcErrEndpoint
      core%reason = 'stock endpoint map or cell images unavailable'
      return
    end if
    nsrc = domain%num_cell_local + domain%num_cell_boundary
    if (nsrc < 0 .or. nsrc > min(size(domain%num_atom),                   &
          size(domain%id_l2g,2), size(domain%cell_l2gx),                  &
          size(domain%cell_l2gy), size(domain%cell_l2gz),                 &
          size(domain%cell_l2gx_orig), size(domain%cell_l2gy_orig),       &
          size(domain%cell_l2gz_orig))) then
      status = GcErrArg
      core%reason = 'endpoint source cell span invalid'
      return
    end if

    allocate(core%endpoint_capture(size(core%terms)))
    do k = 1, size(core%terms)
      ncell = int(core%terms(k)%ncell)
      arity = int(core%terms(k)%arity)
      if (ncell < 0 .or. ncell > size(domain%num_atom) .or. &
          arity < 1 .or. arity > GcMaxTermArityI .or. &
          mod(core%terms(k)%list_cell_stride, int(arity,c_int64_t)) /= 0) then
        status = GcErrArg
        core%reason = 'ordinary term descriptor shape invalid'
        return
      end if
      limit = core%terms(k)%list_cell_stride / int(arity,c_int64_t)
      if (limit < 1 .or. limit > int(huge(maxterm),c_int64_t)) then
        status = GcErrCapacity
        core%reason = 'ordinary term cell stride exceeds local capacity'
        return
      end if
      maxterm = int(limit)
      call c_f_pointer(core%terms(k)%num_term, count, [ncell])
      call c_f_pointer(core%terms(k)%list, list, [arity,maxterm,ncell])
      if (any(count(1:ncell) < 0 .or. count(1:ncell) > maxterm)) then
        status = GcErrArg
        core%reason = 'ordinary term count exceeds source list'
        return
      end if
      ! bounded by the size of the list array itself
      total = sum(int(count(1:ncell),c_int64_t)) * int(arity,c_int64_t)
      if (total == 0) cycle
      allocate(core%endpoint_capture(k)%image(total), &
               core%endpoint_capture(k)%cell(total),  &
               core%endpoint_capture(k)%slot(total),  stat=ierr)
      if (ierr /= 0) then
        status = GcErrNoMem
        core%reason = 'ordinary endpoint capture allocation failed'
        return
      end if
      at = 0_c_int64_t
      cells: do c = 1, ncell
        do t = 1, count(c)
          do e = 1, arity
            at = at + 1_c_int64_t
            status = GcErrEndpoint
            gid = list(e,t,c)
            if (gid < 1 .or. gid > size(domain%id_g2l,2)) exit cells
            sc = int(domain%id_g2l(1,gid))
            ss = int(domain%id_g2l(2,gid))
            if (sc < 1 .or. sc > nsrc .or. ss < 1 .or. &
                ss > size(domain%id_l2g,1)) exit cells
            if (ss > domain%num_atom(sc) .or. &
                domain%id_l2g(ss,sc) /= gid) exit cells
            ext  = [domain%cell_l2gx(sc), domain%cell_l2gy(sc), &
                    domain%cell_l2gz(sc)]
            orig = [domain%cell_l2gx_orig(sc), domain%cell_l2gy_orig(sc), &
                    domain%cell_l2gz_orig(sc)]
            packed = 0_c_int64_t
            do axis = 1, 3
              width = int(core%geometry%cell(axis))
              delta = ext(axis) - orig(axis)
              if (width <= 0) exit cells
              if (mod(delta,width) /= 0) exit cells
              shift = delta / width
              status = GcErrCapacity
              if (shift < -32768 .or. shift > 32767) exit cells
              status = GcErrEndpoint
              packed = ior(packed, ishft(iand(int(shift,c_int64_t), &
                          int(z'ffff',c_int64_t)), 16*(axis-1)))
            end do
            status = GcOk
            core%endpoint_capture(k)%image(at) = packed
            core%endpoint_capture(k)%cell(at) = int(sc-1,c_int32_t)
            core%endpoint_capture(k)%slot(at) = int(ss-1,c_int32_t)
          end do
        end do
      end do cells
      if (status /= GcOk) then
        write(core%reason,'(A,I0,A,I0,A,I0)') &
          'endpoint source invalid kind=', core%terms(k)%kind, &
          ' cell=', c, ' term=', t
        return
      end if
      core%terms(k)%endpoint_count = total
      core%terms(k)%endpoint_image = c_loc(core%endpoint_capture(k)%image(1))
      core%terms(k)%endpoint_cell  = c_loc(core%endpoint_capture(k)%cell(1))
      core%terms(k)%endpoint_slot  = c_loc(core%endpoint_capture(k)%slot(1))
    end do

  end subroutine gpu_core_capture_endpoints

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_describe
  !> @brief        fill the POD descriptors from the frozen GENESIS state
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_describe(core, domain, enefunc, constraints, boundary, &
                               nbupdate_period, my_rank, nproc, replica)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),   target, intent(in)    :: domain
    type(s_enefunc),  target, intent(in)    :: enefunc
    type(s_constraints), target, intent(in) :: constraints
    type(s_boundary),         intent(in)    :: boundary
    integer,                  intent(in)    :: nbupdate_period
    integer,                  intent(in)    :: my_rank, nproc, replica

    integer                  :: ncell, max_atom

    ncell    = domain%num_cell_local + domain%num_cell_boundary
    max_atom = size(domain%id_l2g, 1)

    ! geometry
    core%geometry = s_gc_geometry()
    core%geometry%cell       = int([boundary%num_cells_x,                 &
                                    boundary%num_cells_y,                 &
                                    boundary%num_cells_z], c_int32_t)
    core%geometry%num_domain = int(boundary%num_domain(1:3), c_int32_t)
    core%geometry%cell_start = int(domain%cell_start(1:3),   c_int32_t)
    core%geometry%cell_end   = int(domain%cell_end(1:3),     c_int32_t)
    core%geometry%system_size = real(domain%system_size(1:3), c_double)
    core%geometry%cell_size   = real(domain%cell_size(1:3),   c_double)
    ! the origin GENESIS's own cell assignment subtracts before it wraps
    core%geometry%origin = real([boundary%origin_x, boundary%origin_y,    &
                                 boundary%origin_z], c_double)
    core%geometry%ncell_local    = int(domain%num_cell_local,    c_int32_t)
    core%geometry%ncell_boundary = int(domain%num_cell_boundary, c_int32_t)
    core%geometry%pairlistdist   = real(enefunc%pairlistdist, c_double)
    core%geometry%cutoffdist     = real(enefunc%cutoffdist,   c_double)
    core%geometry%table_support_radius = &
        real(gpu_core_support_radius(enefunc), c_double)
    ! The real-space kernel's zero-force prune skips a pair only beyond
    ! cutoff2*density/(density-2), one row more generous than the table's
    ! support; the cubic cutoff and plain-LJ kernels prune at the support.
    if (.not. enefunc%pme_use .or. enefunc%vdw == VDWCutoff) then
      core%geometry%prune_support_radius = core%geometry%table_support_radius
    else if (enefunc%table%density > 3.0_wp) then
      core%geometry%prune_support_radius =                                &
          real(sqrt(enefunc%cutoffdist * enefunc%cutoffdist               &
                    * enefunc%table%density                               &
                    / (enefunc%table%density - 2.0_wp)), c_double)
    end if
    core%geometry%nbupdate_period = int(nbupdate_period, c_int32_t)
    ! FP64 is the only host precision; the C side declines otherwise
    if (wp == dp .and. wip == dp) core%geometry%host_fp64 = 1_c_int32_t

    ! cells
    call gpu_core_build_tie_key(core, domain)
    core%cells = s_gc_cell()
    core%cells%ncell          = int(ncell, c_int64_t)
    core%cells%cell_l2gx      = c_loc(domain%cell_l2gx(1))
    core%cells%cell_l2gy      = c_loc(domain%cell_l2gy(1))
    core%cells%cell_l2gz      = c_loc(domain%cell_l2gz(1))
    core%cells%cell_l2gx_orig = c_loc(domain%cell_l2gx_orig(1))
    core%cells%cell_l2gy_orig = c_loc(domain%cell_l2gy_orig(1))
    core%cells%cell_l2gz_orig = c_loc(domain%cell_l2gz_orig(1))
    core%cells%cell_tie_key   = c_loc(core%tie_key(1))

    ! atoms
    core%atoms = s_gc_atom()
    core%atoms%ncell           = int(ncell, c_int64_t)
    core%atoms%num_atom        = c_loc(domain%num_atom(1))
    core%atoms%gid_cell_stride = int(max_atom, c_int64_t)
    core%atoms%xyz_cell_stride = int(3*max_atom, c_int64_t)
    core%atoms%gid             = c_loc(domain%id_l2g(1,1))
    core%atoms%coord           = c_loc(domain%coord(1,1,1))
    core%atoms%velocity        = c_loc(domain%velocity(1,1,1))
    core%atoms%charge          = c_loc(domain%charge(1,1))
    core%atoms%mass            = c_loc(domain%mass(1,1))
    core%atoms%inv_mass        = c_loc(domain%inv_mass(1,1))
    core%atoms%atom_cls        = c_loc(domain%atom_cls_no(1,1))

    ! groups: waters and the rigid hydrogen groups
    core%groups = s_gc_group()
    core%groups%ncell = int(ncell, c_int64_t)
    if (allocated(domain%water_list) .and. allocated(domain%num_water)) then
      core%groups%water_atom_count  = int(size(domain%water_list,1), c_int32_t)
      core%groups%num_water         = c_loc(domain%num_water(1))
      core%groups%water_list        = c_loc(domain%water_list(1,1,1))
      core%groups%water_cell_stride = int(size(domain%water_list,1) &
                                        * size(domain%water_list,2), c_int64_t)
    end if
    if (allocated(constraints%HGr_local) .and. &
        allocated(constraints%HGr_bond_list)) then
      core%groups%hgr_max_h = int(size(constraints%HGr_local,1), c_int32_t)
      core%groups%hgr_local = c_loc(constraints%HGr_local(1,1))
      core%groups%hgr_local_h_stride    = 1_c_int64_t
      core%groups%hgr_local_cell_stride = &
          int(size(constraints%HGr_local,1), c_int64_t)
      core%groups%hgr_bond_list = c_loc(constraints%HGr_bond_list(1,1,1,1))
      core%groups%hgr_list_member_stride = 1_c_int64_t
      core%groups%hgr_list_group_stride  = &
          int(size(constraints%HGr_bond_list,1), c_int64_t)
      core%groups%hgr_list_h_stride = core%groups%hgr_list_group_stride &
          * int(size(constraints%HGr_bond_list,2), c_int64_t)
      core%groups%hgr_list_cell_stride = core%groups%hgr_list_h_stride  &
          * int(size(constraints%HGr_bond_list,3), c_int64_t)
    end if

    ! canonical terms: the five bonded kinds whose per-cell lists hold global
    ! atom indices and whose parameters are stored one array per field
    if (allocated(core%terms)) deallocate(core%terms)
    allocate(core%terms(0))

    if (allocated(enefunc%num_bond) .and. allocated(enefunc%bond_list))   &
      call gpu_core_add_term(core, GcTermBond, 2, enefunc%num_bond,       &
             enefunc%bond_list, size(enefunc%bond_force_const,1),         &
             pr1=enefunc%bond_force_const, pr2=enefunc%bond_dist_min,     &
             pbc1=enefunc%bond_pbc)

    if (allocated(enefunc%num_angle) .and. allocated(enefunc%angle_list)) &
      call gpu_core_add_term(core, GcTermAngle, 3, enefunc%num_angle,     &
             enefunc%angle_list, size(enefunc%angle_force_const,1),       &
             pr1=enefunc%angle_force_const, pr2=enefunc%angle_theta_min,  &
             pr3=enefunc%urey_force_const,  pr4=enefunc%urey_rmin,        &
             pbcn=enefunc%angle_pbc)

    if (allocated(enefunc%num_dihedral) .and. allocated(enefunc%dihe_list)) &
      call gpu_core_add_term(core, GcTermDihedral, 4, enefunc%num_dihedral, &
             enefunc%dihe_list, size(enefunc%dihe_force_const,1),         &
             pr1=enefunc%dihe_force_const, pr2=enefunc%dihe_phase,        &
             pi1=enefunc%dihe_periodicity, pbcn=enefunc%dihe_pbc)

    if (allocated(enefunc%num_improper) .and. allocated(enefunc%impr_list)) &
      call gpu_core_add_term(core, GcTermImproper, 4, enefunc%num_improper, &
             enefunc%impr_list, size(enefunc%impr_force_const,1),         &
             pr1=enefunc%impr_force_const, pr2=enefunc%impr_phase,        &
             pi1=enefunc%impr_periodicity, pbcn=enefunc%impr_pbc)

    if (allocated(enefunc%num_cmap) .and. allocated(enefunc%cmap_list))   &
      call gpu_core_add_term(core, GcTermCmap, 8, enefunc%num_cmap,       &
             enefunc%cmap_list, size(enefunc%cmap_type,1),                &
             pi1=enefunc%cmap_type, pbcn=enefunc%cmap_pbc)

    ! The 1-4 and excluded pairs, stored per CELL PAIR (two cells and two
    ! cell-local slots per endpoint; the outer index is the assigned cell)
    core%state = s_gc_state()
    if (allocated(enefunc%num_nb14_calc) .and. &
        allocated(enefunc%nb14_calc_list)) &
      call add_pairterm(GcTermNb14, enefunc%num_nb14_calc, &
                        enefunc%nb14_calc_list)
    if (allocated(enefunc%num_nonb_excl) .and. &
        allocated(enefunc%nonb_excl_list)) &
      call add_pairterm(GcTermExcl, enefunc%num_nonb_excl, &
                        enefunc%nonb_excl_list)

    call gpu_core_describe_forcefield(core, enefunc)

    if (core%state%num_pairterms > 0_c_int32_t) &
      core%state%pairterms = c_loc(core%pairterms(1))
    core%state%geometry  = c_loc(core%geometry)
    core%state%cells     = c_loc(core%cells)
    core%state%atoms     = c_loc(core%atoms)
    core%state%groups    = c_loc(core%groups)
    core%state%num_terms = int(size(core%terms), c_int32_t)
    if (size(core%terms) > 0) core%state%terms = c_loc(core%terms(1))
    core%state%rank    = int(my_rank, c_int32_t)
    core%state%nproc   = int(nproc,   c_int32_t)
    core%state%replica = int(replica, c_int32_t)
    core%state%comm    = int(mpi_comm_country, c_int32_t)

  contains

    subroutine add_pairterm(kind, num, list)
      integer(c_int32_t),      intent(in) :: kind
      integer,         target, intent(in) :: num(:), list(:,:,:)
      integer :: i
      i = int(core%state%num_pairterms) + 1
      core%pairterms(i) = s_gc_pairterm()
      core%pairterms(i)%kind             = kind
      core%pairterms(i)%ncell_local      = int(domain%num_cell_local, &
                                               c_int64_t)
      core%pairterms(i)%num_term         = c_loc(num(1))
      core%pairterms(i)%list             = c_loc(list(1,1,1))
      core%pairterms(i)%list_term_stride = 4_c_int64_t
      core%pairterms(i)%list_cell_stride = int(size(list,1)*size(list,2), &
                                               c_int64_t)
      core%state%num_pairterms = int(i, c_int32_t)
    end subroutine add_pairterm

  end subroutine gpu_core_describe

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_describe_forcefield
  !> @brief        the force-field forms the native kernels read: the table
  !!               form, the improper and 1-4 forms stock evaluates
  !!               (compute_energy_charmm, _amber, _gro_amber), GENESIS's
  !!               periodicity encoding and the per-pair 1-4 scales
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_describe_forcefield(core, enefunc)

    type(s_gpu_core),        intent(inout) :: core
    type(s_enefunc), target, intent(in)    :: enefunc

    integer                  :: i

    core%tables = s_gc_table()
    core%tables%table_form = GcTablePmeLinear
    if (enefunc%vdw == VDWCutoff) core%tables%table_form = GcTablePmeElec
    if (.not. enefunc%pme_use) core%tables%table_form = GcTableCutoffCubic

    ! CHARMM: harmonic impropers, the switched 1-4 LJ from its own pair
    ! table, and no per-pair scale
    core%tables%improper_form   = GcImproperHarmonic
    core%tables%nb14_form       = GcNb14Table
    core%tables%periodicity_mod = 0_c_int32_t

    if (enefunc%forcefield == ForcefieldAMBER .or. &
        enefunc%forcefield == ForcefieldGROAMBER) then
      if (enefunc%forcefield == ForcefieldAMBER) &
        core%tables%improper_form = GcImproperFourier
      core%tables%nb14_form       = GcNb14Scaled
      core%tables%periodicity_mod = int(enefunc%notation_14types, c_int32_t)
      do i = 1, int(core%state%num_pairterms)
        if (core%pairterms(i)%kind /= GcTermNb14) cycle
        core%pairterms(i)%qq_scale = c_loc(enefunc%nb14_qq_scale(1,1))
        core%pairterms(i)%lj_scale = c_loc(enefunc%nb14_lj_scale(1,1))
        core%pairterms(i)%scale_cell_stride = &
            int(size(enefunc%nb14_qq_scale,1), c_int64_t)
      end do
    end if

  end subroutine gpu_core_describe_forcefield

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_setup
  !> @brief        describe the frozen state, admit it and import it
  !! @param[out]   status : GcOk, or the refusal that stopped it
  !
  !  Every phase is voted over the simulation communicator, so the status
  !  returned is the same on every rank: a rank that declines cannot go on
  !  on the CPU while its peers enter GPU collectives.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_setup(core, domain, enefunc, constraints, boundary, &
                            nbupdate_period, my_rank, nproc, replica, status)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),   target, intent(in)    :: domain
    type(s_enefunc),  target, intent(in)    :: enefunc
    type(s_constraints), target, intent(in) :: constraints
    type(s_boundary),         intent(in)    :: boundary
    integer,                  intent(in)    :: nbupdate_period
    integer,                  intent(in)    :: my_rank, nproc, replica
    integer,                  intent(out)   :: status

    type(c_ptr)              :: reason

    core%active = .false.
    core%ctx    = c_null_ptr

    call gpu_core_describe(core, domain, enefunc, constraints, boundary, &
                           nbupdate_period, my_rank, nproc, replica)
    call gpu_core_capture_endpoints(core, domain, status)
    call gpu_core_vote(status, nproc)
    if (status /= GcOk) return

    reason = c_null_ptr
    status = gpu_core_try_setup(c_loc(core%state), reason)
    core%reason = gc_c_string(reason)
    call gpu_core_vote(status, nproc)
    if (status /= GcOk) return

    status = gpu_core_create(c_loc(core%state), core%ctx)
    call gpu_core_vote(status, nproc)
    if (status /= GcOk) then
      write(core%reason,'(A,I0)') 'gpu_core_create refused with status ', status
      return
    end if

    status = gpu_core_import(core%ctx, c_loc(core%state))
    call gpu_core_vote(status, nproc)
    if (status /= GcOk) then
      write(core%reason,'(A,I0)') 'gpu_core_import refused with status ', status
      return
    end if

    core%active = .true.

  end subroutine gpu_core_setup

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_admit_real_mask
  !> @brief        import the stock real-space zero bits into the context,
  !!               then read back the allocation ledger and the guard
  !
  !  The final stock mask includes neutral 1-2/1-3/1-4 pairs; nonb_excl_list
  !  is charge gated for the reciprocal correction and cannot stand in for
  !  these zero bits. Call once after the import, before the first rebuild.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_admit_real_mask(core, domain, enefunc, status)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),   target, intent(in)    :: domain
    type(s_enefunc),  target, intent(in)    :: enefunc
    integer,                  intent(out)   :: status

    type(c_ptr)              :: near_list, near_mask
    integer                  :: max_atom

    status = GcErrState
    if (.not. core%active) return

    if (.not. allocated(enefunc%exclusion_mask1) .or. &
        .not. allocated(enefunc%exclusion_mask) .or.  &
        .not. allocated(domain%cell_pairlist1)) then
      core%reason = 'stock real-space exclusion mask unavailable'
      status = GcErrUnsupport
      return
    end if
    max_atom = size(domain%id_l2g,1)
    if (size(enefunc%exclusion_mask1,1) /= max_atom .or. &
        size(enefunc%exclusion_mask1,2) /= max_atom .or. &
        size(enefunc%exclusion_mask,1)  /= max_atom .or. &
        size(enefunc%exclusion_mask,2)  /= max_atom .or. &
        size(enefunc%exclusion_mask1,3) < domain%num_cell_local .or. &
        size(domain%cell_pairlist1,2) < size(enefunc%exclusion_mask,3)) then
      core%reason = 'stock real-space exclusion mask shape invalid'
      status = GcErrArity
      return
    end if
    near_list = c_null_ptr
    near_mask = c_null_ptr
    if (size(enefunc%exclusion_mask,3) > 0) then
      near_list = c_loc(domain%cell_pairlist1(1,1))
      near_mask = c_loc(enefunc%exclusion_mask(1,1,1))
    end if
    status = gpu_core_import_real_mask(core%ctx,                          &
             int(domain%num_cell_local,c_int64_t), int(max_atom,c_int64_t),&
             c_loc(domain%num_atom(1)), c_loc(domain%id_l2g(1,1)),        &
             int(size(enefunc%exclusion_mask,3),c_int64_t),               &
             near_list, c_loc(enefunc%exclusion_mask1(1,1,1)), near_mask)
    if (status /= GcOk) then
      core%reason = 'stock real-space exclusion mask refused by the core'
      return
    end if

    status = gpu_core_counts(core%ctx, core%num_owned, core%num_halo)
    if (status /= GcOk) return
    status = gpu_core_list_guard_plan(core%ctx, core%list_radius,          &
                                      core%support_radius, core%half_skin)

  end subroutine gpu_core_admit_real_mask

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_attach_posres
  !> @brief        hand this rank's positional restraints to the core, as
  !!               compute_energy_restraints_pos reads them; collective, every
  !!               rank calls, with none when the run has no restraint
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_attach_posres(core, domain, enefunc, status)

    type(s_gpu_core),         intent(inout) :: core
    type(s_domain),           intent(in)    :: domain
    type(s_enefunc),          intent(in)    :: enefunc
    integer,                  intent(out)   :: status

    integer(c_int32_t), allocatable, target :: gid(:)
    real(c_double),     allocatable, target :: ref(:,:), par(:,:)
    integer                  :: c, ix, n

    n = 0
    if (enefunc%restraint_posi) &
      n = sum(enefunc%num_restraint(1:domain%num_cell_local))
    allocate(gid(max(n,1)), ref(3,max(n,1)), par(4,max(n,1)))
    n = 0
    if (enefunc%restraint_posi) then
      do c = 1, domain%num_cell_local
        do ix = 1, enefunc%num_restraint(c)
          n = n + 1
          gid(n)     = enefunc%restraint_atom(ix,c)
          ref(1:3,n) = enefunc%restraint_coord(1:3,ix,c)
          par(1:4,n) = enefunc%restraint_force(1:4,ix,c)
        end do
      end do
    end if
    status = gpu_core_posres_set(core%ctx, int(n,c_int64_t), c_loc(gid(1)), &
                                 c_loc(ref(1,1)), c_loc(par(1,1)))
    if (status /= GcOk) &
      core%reason = 'positional restraints refused by the core'

  end subroutine gpu_core_attach_posres

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_finalize
  !> @brief        release the context and its device memory
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_finalize(core)

    type(s_gpu_core),        intent(inout) :: core

    integer                  :: status

    if (c_associated(core%ctx)) then
      status = gpu_core_destroy(core%ctx)
      if (status /= GcOk) &
        write(MsgOut,'(A,A)') 'Setup_GPU_Core> destroy refused: ', &
                              trim(gc_status_text(status))
    end if
    core%ctx    = c_null_ptr
    core%active = .false.
    if (allocated(core%terms))            deallocate(core%terms)
    if (allocated(core%endpoint_capture)) deallocate(core%endpoint_capture)
    if (allocated(core%tie_key))          deallocate(core%tie_key)

  end subroutine gpu_core_finalize

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_build_rigid
  !> @brief        flatten GENESIS's hydrogen groups by representative id
  !
  !  GENESIS addresses HGr_bond_dist by (member, group, hydrogen count,
  !  cell), an addressing the native group sort destroys; the records are
  !  keyed by the heavy atom's global id and sorted (heapsort, in place), so
  !  the native side finds a group's distances by binary search.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_build_rigid(core, domain, constraints)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),          intent(in)    :: domain
    type(s_constraints),     intent(in)    :: constraints

    integer                  :: icel, j, ig, n, root, last
    integer, parameter       :: h = GcMaxHGroupH
    real(c_double)           :: d(h)

    if (allocated(core%rigid_gid))   deallocate(core%rigid_gid)
    if (allocated(core%rigid_arity)) deallocate(core%rigid_arity)
    if (allocated(core%rigid_dist))  deallocate(core%rigid_dist)

    if (.not. allocated(constraints%HGr_local)) return
    if (.not. allocated(constraints%HGr_bond_dist)) return

    n = sum(constraints%HGr_local(:, 1:domain%num_cell_local))
    if (n == 0) return

    allocate(core%rigid_gid(n), core%rigid_arity(n), core%rigid_dist(h*n))
    core%rigid_dist(1:h*n) = 0.0_c_double

    n = 0
    do icel = 1, domain%num_cell_local
      do j = 1, size(constraints%HGr_local, 1)
        do ig = 1, constraints%HGr_local(j, icel)
          n = n + 1
          core%rigid_gid(n)   = int(domain%id_l2g(                       &
              constraints%HGr_bond_list(1, ig, j, icel), icel), c_int64_t)
          core%rigid_arity(n) = int(j, c_int32_t)
          core%rigid_dist(h*(n-1)+1:h*(n-1)+j) = &
              real(constraints%HGr_bond_dist(2:j+1, ig, j, icel), c_double)
        end do
      end do
    end do

    ! heapsort by rigid_gid: build the heap, then move each maximum last
    do root = n/2, 1, -1
      call sift(root, n)
    end do
    do last = n, 2, -1
      call swap(1, last)
      call sift(1, last-1)
    end do

  contains

    subroutine sift(start, m)
      integer, intent(in) :: start, m
      integer :: r, c
      r = start
      do while (2*r <= m)
        c = 2*r
        if (c < m) then
          if (core%rigid_gid(c) < core%rigid_gid(c+1)) c = c+1
        end if
        if (core%rigid_gid(r) >= core%rigid_gid(c)) exit
        call swap(r, c)
        r = c
      end do
    end subroutine sift

    subroutine swap(a, b)
      integer, intent(in) :: a, b
      core%rigid_gid([a,b])   = core%rigid_gid([b,a])
      core%rigid_arity([a,b]) = core%rigid_arity([b,a])
      d(1:h) = core%rigid_dist(h*(a-1)+1:h*a)
      core%rigid_dist(h*(a-1)+1:h*a) = core%rigid_dist(h*(b-1)+1:h*b)
      core%rigid_dist(h*(b-1)+1:h*b) = d(1:h)
    end subroutine swap

  end subroutine gpu_core_build_rigid

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_attach_step
  !> @brief        hand the native core its tables, its rigid groups and its
  !!               integrator plan, all static per run, and ask whether it
  !!               accepts them
  !! @param[in]    ens_kind   : GcEnsembleNVE, GcEnsembleNVT or GcEnsembleNPT
  !! @param[in]    thermostat : GcThermo*
  !! @param[in]    group_tp   : 1 when the group temperature convention is on
  !! @param[in]    nbupdate   : the pair-list update period
  !! @param[in]    thermo     : the thermostat period
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_attach_step(core, domain, enefunc, constraints,      &
                                  ens_kind, thermostat, group_tp, dt,      &
                                  nbupdate, thermo, status)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),   target, intent(in)    :: domain
    type(s_enefunc),  target, intent(in)    :: enefunc
    type(s_constraints), target, intent(in) :: constraints
    integer,                  intent(in)    :: ens_kind, thermostat, group_tp
    real(dp),                 intent(in)    :: dt
    integer,                  intent(in)    :: nbupdate, thermo
    integer,                  intent(out)   :: status

    type(c_ptr)              :: rgid, rarity, rdist, ppme, reason
    integer(c_int64_t)       :: n, nrigid

    status = GcErrState
    if (.not. core%active) then
      core%reason = 'no active native context for the step plan'
      return
    end if

    ! the static tables (the forms are gpu_core_describe_forcefield's)
    core%tables%num_atom_cls = int(enefunc%num_atom_cls, c_int32_t)
    core%tables%cutoff_int   = int(enefunc%table%cutoff_int, c_int32_t)
    core%tables%density      = real(enefunc%table%density, c_double)
    core%tables%cutoffdist   = real(enefunc%cutoffdist, c_double)
    core%tables%nonb_lj12    = c_loc(enefunc%nonb_lj12(1,1))
    core%tables%nonb_lj6     = c_loc(enefunc%nonb_lj6(1,1))
    core%tables%nb14_lj12    = c_loc(enefunc%nb14_lj12(1,1))
    core%tables%nb14_lj6     = c_loc(enefunc%nb14_lj6(1,1))
    core%tables%table_ene    = c_loc(enefunc%table%table_ene(1))
    core%tables%table_grad   = c_loc(enefunc%table%table_grad(1))
    ! electrostatic = CUTOFF: the cubic cutoff table has no correction table
    if (enefunc%pme_use) then
      core%tables%table_ecor  = c_loc(enefunc%table%table_ecor(1))
      core%tables%table_decor = c_loc(enefunc%table%table_decor(1))
    end if
    core%tables%nonbond_precision = GcNonbondDouble
    if (enefunc%nonbond_precision == NonbondPrecisionMixed) &
      core%tables%nonbond_precision = GcNonbondMixed
    core%tables%ewald_evaluation = GcEwaldAuto
    if (enefunc%ewald_evaluation == EwaldEvaluationTable) &
      core%tables%ewald_evaluation = GcEwaldTable
    if (enefunc%ewald_evaluation == EwaldEvaluationAnalytic) &
      core%tables%ewald_evaluation = GcEwaldAnalytic
    ! flexible water: stock evaluates these terms from scalars over
    ! water_list (sp_energy_bonds.fpp, sp_energy_angles.fpp)
    if (enefunc%table%water_bond_calc) then
      core%tables%water_bond_calc = 1_c_int32_t
      core%tables%water_oh_bond   = real(enefunc%table%OH_bond,  c_double)
      core%tables%water_oh_force  = real(enefunc%table%OH_force, c_double)
      if (enefunc%table%water_bond_calc_HH) then
        core%tables%water_bond_hh  = 1_c_int32_t
        core%tables%water_hh_bond  = real(enefunc%table%HH_bond,  c_double)
        core%tables%water_hh_force = real(enefunc%table%HH_force, c_double)
      end if
    end if
    if (enefunc%table%water_angle_calc) then
      core%tables%water_angle_calc = 1_c_int32_t
      core%tables%water_hoh_angle  = real(enefunc%table%HOH_angle, c_double)
      core%tables%water_hoh_force  = real(enefunc%table%HOH_force, c_double)
    end if
    if (allocated(enefunc%cmap_coef) .and. &
        allocated(enefunc%cmap_resolution)) then
      core%tables%cmap_ntype = int(size(enefunc%cmap_coef,5), c_int32_t)
      core%tables%cmap_ngrid = int(size(enefunc%cmap_coef,3), c_int32_t)
      core%tables%cmap_resolution = c_loc(enefunc%cmap_resolution(1))
      core%tables%cmap_coef       = c_loc(enefunc%cmap_coef(1,1,1,1,1))
    end if

    ! the constraint parameters
    core%cons = s_gc_constraint()
    core%cons%rigid_bond      = merge(1_c_int32_t, 0_c_int32_t, &
                                      constraints%rigid_bond)
    core%cons%fast_water      = merge(1_c_int32_t, 0_c_int32_t, &
                                      constraints%fast_water)
    core%cons%shake_iteration = int(constraints%shake_iteration, c_int32_t)
    core%cons%shake_tolerance = real(constraints%shake_tolerance, c_double)
    core%cons%water_mass_o    = real(constraints%water_massO, c_double)
    core%cons%water_mass_h    = real(constraints%water_massH, c_double)
    core%cons%water_r_oh      = real(constraints%water_rOH, c_double)
    core%cons%water_r_hh      = real(constraints%water_rHH, c_double)

    call gpu_core_build_rigid(core, domain, constraints)
    nrigid = 0_c_int64_t
    rgid   = c_null_ptr
    rarity = c_null_ptr
    rdist  = c_null_ptr
    if (allocated(core%rigid_gid)) then
      nrigid = int(size(core%rigid_gid), c_int64_t)
      rgid   = c_loc(core%rigid_gid(1))
      rarity = c_loc(core%rigid_arity(1))
      rdist  = c_loc(core%rigid_dist(1))
    end if

    ! the reciprocal plan, on the mesh stock's PME setup settled on (the
    ! scheme may enlarge an axis); electrostatic = CUTOFF attaches none
    core%pme%ngrid(1:3)    = int(pme_ngrid_used(), c_int32_t)
    core%pme%n_bspline     = int(enefunc%pme_nspline, c_int32_t)
    core%pme%alpha         = real(enefunc%pme_alpha, c_double)
    core%pme%dielec_const  = real(enefunc%dielec_const, c_double)
    core%pme%elecoef       = real(ELECOEF, c_double)
    ppme = c_null_ptr
    if (enefunc%pme_use) ppme = c_loc(core%pme)

    status = int(gpu_core_attach_tables(core%ctx, c_loc(core%tables),  &
                                        c_loc(core%cons), nrigid, rgid,  &
                                        rarity, rdist, ppme))
    if (status /= GcOk) then
      core%reason = 'force-field tables or PME plan refused: ' //         &
                    trim(gc_status_text(status))
      return
    end if

    ! the integrator plan
    core%plan%ensemble    = int(ens_kind,   c_int32_t)
    core%plan%thermostat  = int(thermostat, c_int32_t)
    core%plan%group_tp    = int(group_tp,   c_int32_t)
    core%plan%rigid_bond  = core%cons%rigid_bond
    core%plan%dt          = real(dt, c_double)
    core%plan%half_dt     = real(0.5_dp*dt, c_double)
    core%plan%nbupdate_period = int(nbupdate, c_int32_t)
    core%plan%thermo_period   = int(thermo,   c_int32_t)

    reason = c_null_ptr
    status = int(gpu_core_step_setup(core%ctx, c_loc(core%plan), reason))
    if (status /= GcOk) then
      core%reason = 'step plan refused: ' // trim(gc_status_text(status))
      return
    end if

    ! The owned count comes from the context: every boundary transfer is
    ! sized and checked against it.
    status = int(gpu_core_counts(core%ctx, core%num_owned, core%num_halo))
    if (status /= GcOk) then
      core%reason = 'context counts refused: ' // trim(gc_status_text(status))
      return
    end if
    n = core%num_owned
    if (n <= 0_c_int64_t) then
      core%reason = 'the context owns no atom'
      status = GcErrState
      return
    end if

    if (allocated(core%out_gid))   deallocate(core%out_gid)
    if (allocated(core%out_cell))  deallocate(core%out_cell)
    if (allocated(core%out_slot))  deallocate(core%out_slot)
    if (allocated(core%out_coord)) deallocate(core%out_coord)
    if (allocated(core%out_vel))   deallocate(core%out_vel)
    allocate(core%out_gid(n), core%out_cell(n), core%out_slot(n), &
             core%out_coord(3*n), core%out_vel(3*n))

    ! the device's own global-id order, so that a push before any pull has
    ! an order to push into
    status = int(gpu_core_pull_state(core%ctx, n, c_loc(core%out_gid(1)), &
                                     c_null_ptr, c_null_ptr))
    if (status /= GcOk) then
      core%reason = 'initial state pull refused: ' //                     &
                    trim(gc_status_text(status))
      return
    end if

    core%step_ready = .true.

  end subroutine gpu_core_attach_step

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_pull
  !> @brief        bring owned coordinates and velocities back to the host
  !! @param[in]    with_final : also force and velocity_half, for stock's
  !!                            final VV1 (default .false.)
  !
  !  The declared boundary exceptions of doc/21_GPU_Native.rst: output,
  !  restart, replica exchange and the end of the run. The scatter matches
  !  all occupied owned id_l2g slots by global id, including water absent
  !  from GENESIS's solute-only id_g2l.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_pull(core, domain, status, with_final)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),   target, intent(inout) :: domain
    integer,                  intent(out)   :: status
    logical, optional,        intent(in)    :: with_final

    integer                  :: i, icel, ix
    integer(c_int64_t)       :: n, g
    logical                  :: fin

    fin = .false.
    if (present(with_final)) fin = with_final

    status = GcErrState
    if (.not. core%step_ready) return
    if (core%state%nproc > 1) then
      call gpu_core_pull_global(core, domain, fin, status)
      return
    end if
    if (.not. allocated(core%out_gid)) return
    n = core%num_owned
    if (n <= 0) return

    if (fin) then
      if (.not. allocated(core%out_force))    allocate(core%out_force(3*n))
      if (.not. allocated(core%out_vel_half)) allocate(core%out_vel_half(3*n))
      status = int(gpu_core_pull_final_state(core%ctx, n,                 &
                   c_loc(core%out_gid(1)), c_loc(core%out_coord(1)),      &
                   c_loc(core%out_vel(1)), c_loc(core%out_force(1)),      &
                   c_loc(core%out_vel_half(1))))
    else
      status = int(gpu_core_pull_state(core%ctx, n, c_loc(core%out_gid(1)),&
                   c_loc(core%out_coord(1)), c_loc(core%out_vel(1))))
    end if
    if (status /= GcOk) return
    status = int(gpu_core_map_host_slots(n,                               &
                 int(domain%num_cell_local,c_int64_t),                    &
                 int(size(domain%id_l2g,1),c_int64_t),                    &
                 c_loc(domain%num_atom(1)), c_loc(domain%id_l2g(1,1)),    &
                 c_loc(core%out_gid(1)), c_loc(core%out_cell(1)),         &
                 c_loc(core%out_slot(1))))
    if (status /= GcOk) return

    do i = 1, int(n)
      g = core%out_gid(i)
      if (g < 1_c_int64_t .or. g > int(domain%num_atom_all, c_int64_t)) then
        status = GcErrEndpoint
        return
      end if
      icel = core%out_cell(i) + 1
      ix   = core%out_slot(i) + 1
      domain%coord(1:3,ix,icel)    = real(core%out_coord(3*i-2:3*i), wip)
      domain%velocity(1:3,ix,icel) = real(core%out_vel(3*i-2:3*i), wip)
      if (fin) then
        domain%force(1:3,ix,icel) = real(core%out_force(3*i-2:3*i), wip)
        domain%velocity_half(1:3,ix,icel) = &
            real(core%out_vel_half(3*i-2:3*i), wip)
      end if
    end do

  end subroutine gpu_core_pull

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_pull_global
  !> @brief        multi-rank host boundary: device values to the host owners
  !
  !  At more than one rank the device migrates atoms between ranks, while
  !  the host domain keeps the ownership it had at setup.  Each rank pulls
  !  the atoms its device owns now and writes those its host domain holds.
  !  The rest travel through a directory rank chosen by global id block:
  !  device ranks send those rows there, host ranks ask it for the slots
  !  still empty, and it answers in request order.  Every global id must
  !  arrive exactly once; no array is sized by the global atom count.
  !  Collective over mpi_comm_country.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_pull_global(core, domain, with_final, status)

    type(s_gpu_core), target, intent(inout) :: core
    type(s_domain),   target, intent(inout) :: domain
    logical,                  intent(in)    :: with_final
    integer,                  intent(out)   :: status

#ifdef HAVE_MPI_GENESIS
    integer(c_int64_t)              :: n_dev, n_halo
    integer(c_int64_t), allocatable, target :: gid(:)
    integer(c_int32_t), allocatable, target :: cel(:), slt(:)
    real(c_double),     allocatable, target :: crd(:), vel(:), frc(:), vhf(:)
    real(dp),           allocatable :: row(:,:), srow(:,:), rrow(:,:)
    real(dp),           allocatable :: reply(:,:), back(:,:)
    integer,            allocatable :: filled(:,:), at(:)
    integer,            allocatable :: qgid(:), qcel(:), qslt(:), rqgid(:)
    integer,            allocatable :: sgid(:), rsgid(:)
    integer,            allocatable :: nsend(:,:), nrecv(:,:), cur(:)
    integer,            allocatable :: qdsp(:), rqdsp(:), sdsp(:), rsdsp(:)
    integer                         :: nval, n, m, i, j, k, d, g, ierror
    integer                         :: icel, ix, nproc, blk, lo, bad
    integer                         :: nq, nrq, ns, nrs

    nproc  = int(core%state%nproc)
    nval   = merge(12, 6, with_final)

    status = int(gpu_core_counts(core%ctx, n_dev, n_halo))
    if (status == GcOk .and. n_dev < 0_c_int64_t) status = GcErrState
    n = max(int(n_dev), 0)
    m = max(n, 1)
    allocate(gid(m), cel(m), slt(m), crd(3*m), vel(3*m), frc(3*m), vhf(3*m))
    if (status == GcOk .and. n > 0) then
      if (with_final) then
        status = int(gpu_core_pull_final_state(core%ctx, n_dev,          &
                     c_loc(gid(1)), c_loc(crd(1)), c_loc(vel(1)),       &
                     c_loc(frc(1)), c_loc(vhf(1))))
      else
        status = int(gpu_core_pull_state(core%ctx, n_dev, c_loc(gid(1)), &
                     c_loc(crd(1)), c_loc(vel(1))))
      end if
    end if
    if (status == GcOk) &
      status = int(gpu_core_find_host_slots(n_dev,                      &
                   int(domain%num_cell_local,c_int64_t),                &
                   int(size(domain%id_l2g,1),c_int64_t),                &
                   c_loc(domain%num_atom(1)), c_loc(domain%id_l2g(1,1)), &
                   c_loc(gid(1)), c_loc(cel(1)), c_loc(slt(1))))
    call gpu_core_vote(status, nproc)
    if (status /= GcOk) return

    allocate(row(nval,m))
    do i = 1, n
      row(1:3,i) = real(crd(3*i-2:3*i), dp)
      row(4:6,i) = real(vel(3*i-2:3*i), dp)
      if (with_final) then
        row(7:9,i)   = real(frc(3*i-2:3*i), dp)
        row(10:12,i) = real(vhf(3*i-2:3*i), dp)
      end if
    end do

    ! Rows this rank's host domain holds are written here; the others
    ! (strays) and the host slots still empty (requests) are counted per
    ! directory rank.
    bad = 0
    blk = (domain%num_atom_all + nproc - 1) / nproc
    allocate(filled(size(domain%id_l2g,1), domain%num_cell_local))
    allocate(nsend(2,nproc), nrecv(2,nproc))
    filled(:,:) = 0
    nsend(:,:)  = 0
    do i = 1, n
      if (cel(i) >= 0) then
        icel = cel(i) + 1
        ix   = slt(i) + 1
        if (filled(ix,icel) /= 0) bad = 1
        filled(ix,icel) = 1
        call store(icel, ix, row(:,i))
      else
        d = directory(int(gid(i)))
        if (d > 0) nsend(2,d) = nsend(2,d) + 1
      end if
    end do
    do icel = 1, domain%num_cell_local
      do ix = 1, domain%num_atom(icel)
        if (filled(ix,icel) /= 0) cycle
        d = directory(domain%id_l2g(ix,icel))
        if (d > 0) nsend(1,d) = nsend(1,d) + 1
      end do
    end do
    call mpi_alltoall(nsend, 2, mpi_integer, nrecv, 2, mpi_integer, &
                      mpi_comm_country, ierror)

    allocate(qdsp(nproc), rqdsp(nproc), sdsp(nproc), rsdsp(nproc), cur(nproc))
    call prefix(nsend(1,:), qdsp, nq)
    call prefix(nrecv(1,:), rqdsp, nrq)
    call prefix(nsend(2,:), sdsp, ns)
    call prefix(nrecv(2,:), rsdsp, nrs)

    ! Pack in directory order.
    allocate(qgid(max(nq,1)), qcel(max(nq,1)), qslt(max(nq,1)), &
             sgid(max(ns,1)), srow(nval,max(ns,1)))
    cur(:) = qdsp(:)
    do icel = 1, domain%num_cell_local
      do ix = 1, domain%num_atom(icel)
        if (filled(ix,icel) /= 0) cycle
        g = domain%id_l2g(ix,icel)
        d = directory(g)
        if (d <= 0) cycle
        cur(d) = cur(d) + 1
        qgid(cur(d)) = g
        qcel(cur(d)) = icel
        qslt(cur(d)) = ix
      end do
    end do
    cur(:) = sdsp(:)
    do i = 1, n
      if (cel(i) >= 0) cycle
      d = directory(int(gid(i)))
      if (d <= 0) cycle
      cur(d) = cur(d) + 1
      sgid(cur(d))   = int(gid(i))
      srow(:,cur(d)) = row(:,i)
    end do

    allocate(rqgid(max(nrq,1)), rsgid(max(nrs,1)), rrow(nval,max(nrs,1)))
    call mpi_alltoallv(qgid, nsend(1,:), qdsp, mpi_integer,  &
                       rqgid, nrecv(1,:), rqdsp, mpi_integer, &
                       mpi_comm_country, ierror)
    call mpi_alltoallv(sgid, nsend(2,:), sdsp, mpi_integer,  &
                       rsgid, nrecv(2,:), rsdsp, mpi_integer, &
                       mpi_comm_country, ierror)
    call mpi_alltoallv(srow, nval*nsend(2,:), nval*sdsp, mpi_real8,  &
                       rrow, nval*nrecv(2,:), nval*rsdsp, mpi_real8, &
                       mpi_comm_country, ierror)

    ! Directory: match each request to the one stray row of its global id.
    lo = my_country_rank * blk
    allocate(at(max(blk,1)), reply(nval,max(nrq,1)))
    at(:)      = 0
    reply(:,:) = 0.0_dp
    do k = 1, nrs
      j = rsgid(k) - lo
      if (j < 1 .or. j > blk) then
        bad = 1
      else if (at(j) /= 0) then
        bad = 1
      else
        at(j) = k
      end if
    end do
    do k = 1, nrq
      j = rqgid(k) - lo
      if (j < 1 .or. j > blk) then
        bad = 1
      else if (at(j) <= 0) then
        bad = 1
      else
        reply(:,k) = rrow(:,at(j))
        at(j) = -1
      end if
    end do
    if (any(at(:) > 0)) bad = 1

    allocate(back(nval,max(nq,1)))
    call mpi_alltoallv(reply, nval*nrecv(1,:), nval*rqdsp, mpi_real8, &
                       back, nval*nsend(1,:), nval*qdsp, mpi_real8,   &
                       mpi_comm_country, ierror)
    do k = 1, nq
      call store(qcel(k), qslt(k), back(:,k))
    end do

    status = GcOk
    if (bad /= 0) status = GcErrMismatch
    call gpu_core_vote(status, nproc)
#else
    status = GcErrUnsupport
#endif

#ifdef HAVE_MPI_GENESIS
  contains

    ! One-based directory rank of a global id, 0 for an id out of range.
    integer function directory(g)
      integer, intent(in) :: g
      directory = 0
      if (g < 1 .or. g > domain%num_atom_all) then
        bad = 1
      else
        directory = (g - 1) / blk + 1
      end if
    end function directory

    subroutine prefix(cnt, dsp, total)
      integer, intent(in)  :: cnt(:)
      integer, intent(out) :: dsp(:), total
      integer :: r
      total = 0
      do r = 1, size(cnt)
        dsp(r) = total
        total  = total + cnt(r)
      end do
    end subroutine prefix

    subroutine store(icel, ix, v)
      integer,  intent(in) :: icel, ix
      real(dp), intent(in) :: v(:)
      domain%coord(1:3,ix,icel)    = real(v(1:3), wip)
      domain%velocity(1:3,ix,icel) = real(v(4:6), wip)
      if (with_final) then
        domain%force(1:3,ix,icel)         = real(v(7:9), wip)
        domain%velocity_half(1:3,ix,icel) = real(v(10:12), wip)
      end if
    end subroutine store
#endif

  end subroutine gpu_core_pull_global

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_summary
  !> @brief        print the engagement record doc/21_GPU_Native.rst requires:
  !!               a suite that passed through fallback has not validated
  !!               native execution, so the counters are mandatory evidence
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_summary(core, status)

    type(s_gpu_core), target, intent(inout) :: core
    integer,                  intent(out)   :: status

    integer(c_int64_t)       :: scheduled, early, steps, segments

    status = GcErrState
    if (.not. core%active) return

    scheduled = 0_c_int64_t
    early     = 0_c_int64_t
    steps     = 0_c_int64_t
    segments  = 0_c_int64_t
    status = int(gpu_core_step_summary(core%ctx, scheduled, early,   &
                                       steps, segments))
    if (status /= GcOk) return

    if (main_rank) then
      write(MsgOut,'(A,I12,A,I10,A,I10)')                              &
        'GPU_Core_Summary> native_steps=', steps,                      &
        ' scheduled_rebuilds=', scheduled,                             &
        ' early_rebuilds=', early
      write(MsgOut,'(A)')                                              &
        'GPU_Core_Summary> host_particle_bytes_ordinary=0 '//          &
        'host_topology_bytes_ordinary=0'
      write(MsgOut,'(A,I6,A,F10.5)')                                   &
        'GPU_Core_Summary> graph_segments=', segments,                 &
        ' half_skin=', core%half_skin
      if (core%list_guard) then
        write(MsgOut,'(A,F10.5,A,I10)')                                &
          'GPU_Core_Summary> guard_max_displacement=', core%guard_dmax,&
          ' guard_kept_steps_past_buffer=', core%guard_over
      else
        write(MsgOut,'(A,F10.5,A,I10)')                                &
          'GPU_Core_Summary> guard_max_displacement=', core%guard_dmax,&
          ' guard_steps_over_half_skin=', core%guard_over
      end if
    end if

  end subroutine gpu_core_summary

end module sp_gpu_core_mod
