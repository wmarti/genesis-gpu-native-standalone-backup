!--------1---------2---------3---------4---------5---------6---------7---------8
!
!  Module   sp_gpu_core_step_mod
!> @brief   the device-native velocity-Verlet loop
!! @authors GENESIS device-native core
!
!  The native velocity-Verlet loop dispatched from sp_md_vverlet.fpp and
!  sp_md_respa.fpp. The device owns the state for the whole loop; this module
!  owns the host decisions: admission, the thermostat's scalar update
!  (GENESIS's own random stream), the list-validity vote, the rebuild cadence,
!  and the output, restart and replica boundaries (doc/21_GPU_Native.rst), the
!  only places owned coordinates cross back to the host.
!
!--------1---------2---------3---------4---------5---------6---------7---------8

#ifdef HAVE_CONFIG_H
#include "../config.h"
#endif

module sp_gpu_core_step_mod

  use sp_gpu_core_mod
  use sp_gpu_core_abi_mod
  use sp_output_mod
  use sp_dynvars_mod
  use sp_domain_str_mod
  use sp_enefunc_str_mod
  use sp_energy_str_mod,      only: VDWPME
  use sp_pairlist_str_mod
  use sp_constraints_str_mod
  use sp_boundary_str_mod
  use sp_ensemble_str_mod
  use sp_dynvars_str_mod
  use sp_dynamics_str_mod
  use sp_output_str_mod
  use sp_remd_str_mod
  use sp_restraints_str_mod,  only: RestraintsFuncPOSI
  use random_mod
  use math_libs_mod
  use timers_mod
  use messages_mod
  use constants_mod
  use mpi_parallel_mod
  use, intrinsic :: iso_c_binding

#ifdef HAVE_MPI_GENESIS
  use mpi
#endif

  implicit none
  private

  public  :: gpu_core_vverlet
  public  :: gpu_core_remd_active
  public  :: gpu_core_remd_rescale_velocity
  public  :: gpu_core_remd_boundary
  public  :: gpu_core_remd_release

  ! Native context that outlives one vverlet_dynamics call. In a
  ! temperature-REMD run it is created by the first cycle, kept across
  ! perform_replica_exchange (which rescales its velocities in place) and
  ! destroyed by gpu_core_remd_release. Other runs keep the per-call context.
  type(s_gpu_core), save, target :: remd_core
  logical,          save         :: remd_live = .false.
  integer,          save         :: remd_age  = 0

  ! the bonded slots' words of the last collected force evaluation
  real(c_double), save :: exact_words(GcExactNWord) = 0.0_c_double

contains

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Function      gpu_core_admit
  !> @brief        the local half of the accept-or-decline decision
  !! @param[out]   reason : why not, when the answer is no
  !! @return       .true. when every feature of this run is qualified
  !
  !  Everything outside the qualified span declines rather than being
  !  approximated: the integrator, ensemble and thermostat, the potential
  !  terms the native force computes (CHARMM, AMBER and GROAMBER forms of
  !  gpu_core_abi.h, positional restraints only) and, in a REMD run,
  !  temperature exchange only (REUS, gREST, FEP-REMD, pressure and
  !  surface-tension exchange change parameters or need a barostat the
  !  native context does not carry across a cycle).
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  function gpu_core_admit(dynamics, ensemble, constraints, domain, enefunc, &
                          remd, reason) result(ok)

    type(s_dynamics),        intent(in)  :: dynamics
    type(s_ensemble),        intent(in)  :: ensemble
    type(s_constraints),     intent(in)  :: constraints
    type(s_domain),          intent(in)  :: domain
    type(s_enefunc),         intent(in)  :: enefunc
    type(s_remd),            intent(in)  :: remd
    character(*),            intent(out) :: reason
    logical                  :: ok

    logical                  :: npt, respa, remd_run, remd_other
    integer                  :: nlong

    npt   = ensemble%ensemble /= EnsembleNVE .and. &
            ensemble%ensemble /= EnsembleNVT
    respa = dynamics%integrator == IntegratorVRES
    ! a divisor only read under VRES, where it is the input's
    nlong = max(dynamics%elec_long_period, 1)
    remd_run   = allocated(remd%types)
    remd_other = .false.
    if (remd_run) &
      remd_other = any(remd%types(1:remd%dimension) /= RemdTemperature)

    ok = .false.

    if (domain%fep_use) then
      reason = 'FEP is not qualified for the native core'
    else if (ensemble%ensemble /= EnsembleNVE .and.       &
             ensemble%tpcontrol /= TpcontrolBerendsen .and. &
             ensemble%tpcontrol /= TpcontrolBussi     .and. &
             ensemble%tpcontrol /= TpcontrolNHC) then
      reason = 'only Berendsen, Bussi and NHC are native; ' // &
               'the Langevin thermostat and barostat are not'
    else if (npt .and. .not. (ensemble%group_tp .and. constraints%rigid_bond)) &
      then
      reason = 'NPT is native only with group_tp = YES and rigid_bond = YES'
    else if (npt .and. respa .and.                                         &
             (dynamics%baro_period /= dynamics%thermo_period .or.          &
              mod(dynamics%baro_period, nlong) /= 0)) then
      ! r-RESPA's MTK barostat (sp_md_respa.fpp) reads the group virial
      ! from its thermostat tick's force and the virial of an outer step
      reason = 'NPT with integrator VRES is native only when '//          &
               'barostat_period = thermostat_period, a multiple of '//    &
               'elec_long_period'
    else if (dynamics%annealing) then
      reason = 'simulated annealing is not qualified for the native core'
    else if (respa .and. .not. (constraints%rigid_bond .and. &
                                ensemble%group_tp)) then
      ! sp_md_respa.fpp's VV1 needs group temperature under rigid bonds
      reason = 'integrator VRES is native only with rigid_bond = YES '// &
               'and group_tp = YES'
    else if (respa .and.                                                 &
             (mod(dynamics%eneout_period, nlong) /= 0 .or.               &
              mod(dynamics%iend_step - dynamics%istart_step + 1, nlong)  &
              /= 0)) then
      ! energy output and the run end fall on outer steps, where the
      ! reciprocal term of the printed energy is current
      reason = 'integrator VRES is native only when eneout_period '// &
               'and nsteps are multiples of elec_long_period'
    else if (.not. respa .and. dynamics%integrator /= IntegratorVVER) then
      reason = 'only integrators VVER and VRES are native'
    else if (constraints%water_type == TIP4) then
      ! the massless site's force goes back through stock's
      ! water_force_redistribution, which the native step does not have
      reason = 'TIP4P water is not qualified for the native core'
    else if (dynamics%target_md .or. dynamics%steered_md) then
      reason = 'targeted and steered dynamics are not qualified'
    else if (enefunc%forcefield /= ForcefieldCHARMM .and. &
             enefunc%forcefield /= ForcefieldAMBER  .and. &
             enefunc%forcefield /= ForcefieldGROAMBER) then
      reason = 'only the CHARMM, AMBER and GROAMBER force fields are native'
    else if (enefunc%forcefield /= ForcefieldCHARMM .and. &
             .not. enefunc%pme_use) then
      reason = 'AMBER and GROAMBER are native with PME electrostatics only'
    else if (enefunc%forcefield == ForcefieldGROAMBER .and. &
             enefunc%num_rb_dihe_all > 0) then
      reason = 'Ryckaert-Bellemans dihedrals are not computed by the native core'
    else if (enefunc%forcefield == ForcefieldGROAMBER .and. &
             enefunc%num_impr_all > 0) then
      ! compute_energy_gro_amber applies these impropers' forces but never
      ! adds their energy to the total, so no native output could match
      reason = 'GROMACS harmonic impropers (func 2) are not native'
    else if (enefunc%restraint .and. .not. positional_only()) then
      reason = 'only positional restraints are computed by the native core'
    else if (enefunc%restraint .and. enefunc%pressure_position) then
      reason = 'pressure_position is not computed by the native core'
    else if (enefunc%local_restraint .and. local_dihedral_restraint()) then
      ! a [LOCAL_RESTRAINT] dihedral (dihe_kind = 1) takes stock's localres
      ! form; bond and angle restraints are ordinary harmonic terms
      reason = 'local dihedral restraints are not computed by the native core'
    else if (enefunc%vdw == VDWPME) then
      reason = 'LJ-PME (dispersion_pme) is not computed by the native core'
    else if (enefunc%gamd_use) then
      reason = 'GaMD is not computed by the native core'
    else if (enefunc%use_efield) then
      reason = 'an external electric field is not computed by the native core'
    else if (remd_other) then
      reason = 'only temperature REMD is native; REUS, gREST, ' // &
               'pressure and surface-tension exchange are CPU'
    else if (remd_run .and. respa) then
      reason = 'integrator VRES is not native in REMD'
    else if (remd_run .and. npt) then
      reason = 'NPT is not native in REMD'
    else
      reason = 'eligible'
      ok = .true.
    end if

  contains

    ! POSI functions only, and none of the flags that add a term or a
    ! statistic to compute_energy_restraints (RMSD, PC, EM fit, RPATH)
    logical function positional_only()
      integer :: i
      positional_only = .not. (enefunc%restraint_rmsd .or. &
                               enefunc%restraint_pc   .or. &
                               enefunc%restraint_emfit .or. &
                               enefunc%rpath_flag)
      do i = 1, enefunc%num_restraintfuncs
        if (enefunc%restraint_kind(i) /= RestraintsFuncPOSI) &
          positional_only = .false.
      end do
    end function positional_only

    logical function local_dihedral_restraint()
      integer :: c
      local_dihedral_restraint = .false.
      if (.not. allocated(enefunc%dihe_kind) .or. &
          .not. allocated(enefunc%num_dihedral)) return
      do c = 1, min(size(enefunc%num_dihedral), size(enefunc%dihe_kind, 2))
        if (any(enefunc%dihe_kind(1:enefunc%num_dihedral(c), c) /= 0)) then
          local_dihedral_restraint = .true.
          return
        end if
      end do
    end function local_dihedral_restraint

  end function gpu_core_admit

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_decline
  !> @brief        a declined run falls back to stock's loop, except runs
  !!               only the native core evaluates, which stop here rather
  !!               than hide a CPU fallback behind a GPU request
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_decline(enefunc, reason)

    type(s_enefunc),         intent(in)  :: enefunc
    character(*),            intent(in)  :: reason

    if (remd_live) call gpu_core_remd_release_now()
    if (.not. enefunc%pme_use) &
      call gpu_core_abort('Setup_GPU_Core> electrostatic = CUTOFF runs only '// &
                     'on the native core, which declined: ' // trim(reason))
    if (enefunc%nonbond_precision == NonbondPrecisionMixed) &
      call gpu_core_abort('Setup_GPU_Core> nonbond_precision = MIXED runs '// &
                     'only on the native core, which declined: ' // trim(reason))
    if (pme_mesh_native_only) &
      call gpu_core_abort('Setup_GPU_Core> the PME mesh has fewer than '// &
                     'pme_nspline points per cell, which only the native '// &
                     'core runs, and it declined: ' // trim(reason))

  end subroutine gpu_core_decline

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Temperature-REMD entry points used by sp_remd.fpp: whether a context is
  !  resident; an accepted exchange's velocity rescale, applied on the device
  !  (the caller keeps stock's acceptance, random table and host momentum
  !  scaling); the pull before output_remd writes a replica restart
  !  (mod(step, rstout_period) == 0); and the end of run_remd.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  function gpu_core_remd_active() result(active)
    logical :: active
    active = remd_live
  end function gpu_core_remd_active

  subroutine gpu_core_remd_rescale_velocity(factor)

    real(wp),                intent(in)  :: factor

    integer(c_int32_t)       :: verdict
    integer                  :: status

    if (.not. remd_live) &
      call gpu_core_abort('Gpu_Core_Remd> rescale without a resident context')
    verdict = 0_c_int32_t
    status  = int(gpu_core_remd_rescale(remd_core%ctx, &
                  real(factor, c_double), verdict))
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Remd> device velocity rescale failed: ' // &
                     trim(gc_status_text(status)))
    if (verdict /= 0_c_int32_t) &
      call gpu_core_abort('Gpu_Core_Remd> exchange refused on the device: a '// &
                     'velocity or its rescaled value is not finite')

  end subroutine gpu_core_remd_rescale_velocity

  subroutine gpu_core_remd_boundary(dynamics, dynvars, domain)

    type(s_dynamics),        intent(in)    :: dynamics
    type(s_dynvars),         intent(in)    :: dynvars
    type(s_domain),          intent(inout) :: domain

    integer                  :: status

    if (.not. remd_live) return
    if (dynamics%rstout_period <= 0 .or. dynvars%step <= 0) return
    if (mod(dynvars%step, dynamics%rstout_period) /= 0) return
    call gpu_core_pull(remd_core, domain, status)
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Remd> restart boundary pull failed')

  end subroutine gpu_core_remd_boundary

  subroutine gpu_core_remd_release(domain)

    type(s_domain),          intent(inout) :: domain

    integer                  :: status

    if (.not. remd_live) return
    call gpu_core_pull(remd_core, domain, status)
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Remd> final state pull failed')
    call gpu_core_remd_release_now()

  end subroutine gpu_core_remd_release

  subroutine gpu_core_remd_release_now()
    integer :: status
    if (.not. remd_live) return
    call gpu_core_summary(remd_core, status)
    call gpu_core_finalize(remd_core)
    remd_live = .false.
    remd_age  = 0
  end subroutine gpu_core_remd_release_now

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Host thermostat and barostat scalars, ports of the module-private
  !  vel_scale_bussi, vel_scale_berendsen, vel_scale_nhc, update_barostat
  !  (sp_md_vverlet.fpp) and update_barostat_mtk (sp_md_respa.fpp) with the
  !  same arithmetic. Those modules dispatch to this one, so it cannot use
  !  theirs. The random stream is GENESIS's own; the NHC chain stays in
  !  s_dynvars, so a chain's restart is unchanged.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_scale_bussi(degree, dt, tau_t, temp0, ekin, rr, &
                                  scale_vel)

    integer(iintegers),      intent(in)    :: degree
    real(wip),               intent(in)    :: dt, tau_t
    real(wip),               intent(in)    :: temp0
    real(dp),                intent(in)    :: ekin
    real(wip),               intent(in)    :: rr
    real(wip),               intent(inout) :: scale_vel

    real(wip)                :: tempf, tempt, factor

    factor = exp(-dt/tau_t)
    tempf = 2.0_wip * real(ekin,wip)/(real(degree,wip)*KBOLTZ)
    tempt = tempf*factor    &
          + temp0/real(degree,wip)*(1.0_wip-factor)   &
            *(sum_gauss(degree-1)+rr*rr)              &
          + 2.0_wip*sqrt(tempf*temp0/real(degree,wip) &
            *(1.0_wip-factor)*factor)*rr
    scale_vel = sqrt(tempt/tempf)

  end subroutine gpu_core_scale_bussi

  subroutine gpu_core_scale_berendsen(degree, dt, tau_t, temp0, ekin, &
                                      scale_vel)

    integer(iintegers),      intent(in)    :: degree
    real(wip),               intent(in)    :: dt, tau_t
    real(wip),               intent(in)    :: temp0
    real(dp),                intent(in)    :: ekin
    real(wip),               intent(inout) :: scale_vel

    real(wip)                :: tempf

    tempf = 2.0_wip * real(ekin,wip)/(real(degree,wip)*KBOLTZ)
    scale_vel = sqrt(1.0_wip + (dt/tau_t)*(temp0/tempf-1.0_wip))

  end subroutine gpu_core_scale_berendsen

  subroutine gpu_core_scale_nhc(degree, dt, tau_t, temp0, ekin, ensemble, &
                                dynvars, scale_vel)

    integer(iintegers),      intent(in)    :: degree
    real(wip),               intent(in)    :: dt, tau_t
    real(wip),               intent(in)    :: temp0
    real(dp),                intent(in)    :: ekin
    type(s_ensemble), target,intent(inout) :: ensemble
    type(s_dynvars),  target,intent(inout) :: dynvars
    real(wip),               intent(inout) :: scale_vel

    integer                  :: nh_length, nh_step
    integer                  :: i, j, k
    real(dp)                 :: ekf
    real(wip)                :: w(3)
    real(wip)                :: KbT
    real(wip)                :: dt_small, dt_1, dt_2, dt_4
    real(wip)                :: dt_8, scale_kin
    real(wip),       pointer :: nh_mass(:), nh_vel(:)
    real(wip),       pointer :: nh_force(:), nh_coef(:)

    nh_length   = ensemble%nhchain
    nh_step     = ensemble%nhmultistep
    KbT         = KBOLTZ * temp0

    nh_mass     => dynvars%nh_mass
    nh_vel      => dynvars%nh_velocity
    nh_force    => dynvars%nh_force
    nh_coef     => dynvars%nh_coef

    nh_mass(2:nh_length) = KbT * tau_t*tau_t
    nh_mass(1)           = real(degree,wip) * nh_mass(2)

    w(1) = 1.0_dp / (2.0_dp - 2.0_dp**(1.0_dp/3.0_dp))
    w(3) = w(1)
    w(2) = 1.0_dp - w(1) - w(3)

    dt_small  = dt / real(nh_step, wip)
    scale_vel = 1.0_wip
    ekf = 2.0_dp*ekin

    do i = 1, nh_step
      do j = 1, 3

        dt_1 = w(j) * dt_small
        dt_2 = dt_1 * 0.5_dp
        dt_4 = dt_2 * 0.5_dp
        dt_8 = dt_4 * 0.5_dp

        nh_force(nh_length) = nh_mass(nh_length-1) &
                             *nh_vel(nh_length-1)*nh_vel(nh_length-1)-KbT
        nh_force(nh_length) = nh_force(nh_length) / nh_mass(nh_length)
        nh_vel(nh_length) = nh_vel(nh_length) + nh_force(nh_length)*dt_4

        do k = nh_length-1, 2, -1
          nh_force(k) = nh_mass(k-1)*nh_vel(k-1)*nh_vel(k-1)-KbT
          nh_force(k) = nh_force(k) / nh_mass(k)
          nh_coef(k)  = exp(-nh_vel(k+1)*dt_8)
          nh_vel(k)   = nh_vel(k) * nh_coef(k)
          nh_vel(k)   = nh_vel(k) + nh_force(k)*dt_4
          nh_vel(k)   = nh_vel(k) * nh_coef(k)
        end do

        nh_force(1) = real(ekf,wip) - real(degree,wip)*KbT
        nh_force(1) = nh_force(1) / nh_mass(1)
        nh_coef(1)  = exp(-nh_vel(2)*dt_8)
        nh_vel(1)   = nh_vel(1) * nh_coef(1)
        nh_vel(1)   = nh_vel(1) + nh_force(1)*dt_4
        nh_vel(1)   = nh_vel(1) * nh_coef(1)

        scale_kin = exp(-nh_vel(1)*dt_1)
        scale_vel = scale_vel * exp(-nh_vel(1)*dt_2)
        ekf = ekf * scale_kin

        nh_force(1) = (ekf - real(degree,wip)*KbT) / nh_mass(1)
        nh_vel(1)   = nh_vel(1) * nh_coef(1)
        nh_vel(1)   = nh_vel(1) + nh_force(1)*dt_4
        nh_vel(1)   = nh_vel(1) * nh_coef(1)

        do k = 2, nh_length-1
          nh_force(k) = nh_mass(k-1)*nh_vel(k-1)*nh_vel(k-1)-KbT
          nh_force(k) = nh_force(k) / nh_mass(k)
          nh_vel(k)   = nh_vel(k) * nh_coef(k)
          nh_vel(k)   = nh_vel(k) + nh_force(k)*dt_4
          nh_vel(k)   = nh_vel(k) * nh_coef(k)
        end do

        nh_force(nh_length) = nh_mass(nh_length-1) &
                             *nh_vel(nh_length-1)*nh_vel(nh_length-1)-KbT
        nh_force(nh_length) = nh_force(nh_length) / nh_mass(nh_length)
        nh_vel(nh_length) = nh_vel(nh_length) + nh_force(nh_length)*dt_4

      end do
    end do

  end subroutine gpu_core_scale_nhc

  subroutine gpu_core_update_barostat(ensemble, boundary, press, pressxyz,  &
                                      pressxy, press0, volume, d_ndegf,     &
                                      pmass, dt, ekin, bmoment)

    type(s_ensemble),        intent(in)    :: ensemble
    type(s_boundary),        intent(in)    :: boundary
    real(wip),               intent(in)    :: press(:)
    real(wip),               intent(in)    :: pressxyz
    real(wip),               intent(in)    :: pressxy
    real(wip),               intent(in)    :: press0
    real(wip),               intent(in)    :: volume
    real(wip),               intent(in)    :: d_ndegf
    real(wip),               intent(in)    :: pmass
    real(wip),               intent(in)    :: dt
    real(dp),                intent(in)    :: ekin
    real(wip),               intent(inout) :: bmoment(:)

    real(wip)                :: ekin_real, gamma0, pressxy0

    gamma0    =  ensemble%gamma*ATMOS_P*100.0_wip/1.01325_wip
    ekin_real = real(ekin,wip)

    if (ensemble%isotropy == IsotropyISO) then

      bmoment(1) = bmoment(1) + dt*(volume*(pressxyz - press0)  &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
      bmoment(2) = bmoment(1)
      bmoment(3) = bmoment(1)

    else if (ensemble%isotropy == IsotropySEMI_ISO) then

      if (ensemble%ensemble == EnsembleNPT) then
        bmoment(1) = bmoment(1) + dt*(volume*(pressxy - press0)   &
                                + 2.0_wip*ekin_real/d_ndegf)/pmass
        bmoment(2) = bmoment(1)
        bmoment(3) = bmoment(3) + dt*(volume*(press(3) - press0)   &
                                + 2.0_wip*ekin_real/d_ndegf)/pmass
      else if (ensemble%ensemble == EnsembleNPgT) then
        pressxy0 = press0 - gamma0 / boundary%box_size_z
        bmoment(1) = bmoment(1) + dt*(volume*(pressxy - pressxy0) &
                                + 2.0_wip*ekin_real/d_ndegf)/pmass
        bmoment(2) = bmoment(1)
        bmoment(3) = bmoment(3) + dt*(volume*(press(3) - press0)   &
                                + 2.0_wip*ekin_real/d_ndegf)/pmass
      end if

    else if (ensemble%isotropy == IsotropyANISO) then

      if (ensemble%ensemble == EnsembleNPT) then
        bmoment(1:3) = bmoment(1:3) + dt*(volume*(press(1:3) - press0)    &
                                    + 2.0_wip*ekin_real/d_ndegf)/pmass
      else if (ensemble%ensemble == EnsembleNPgT) then
        pressxy0 = press0 - gamma0 / boundary%box_size_z
        bmoment(1:2) = bmoment(1:2) + dt*(volume*(press(1:2) - pressxy0)  &
                                    + 2.0_wip*ekin_real/d_ndegf)/pmass
        bmoment(3)   = bmoment(3) + dt*(volume*(press(3) - press0)   &
                                  + 2.0_wip*ekin_real/d_ndegf)/pmass
      end if

    else if (ensemble%isotropy == IsotropyXY_Fixed) then

      bmoment(1) = 0.0_wip
      bmoment(2) = 0.0_wip
      bmoment(3) = bmoment(3) + dt*(volume*(press(3) - press0) &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass

    end if

  end subroutine gpu_core_update_barostat

  subroutine gpu_core_update_barostat_mtk(ensemble, press, pressxyz,       &
                                          pressxy, press0, volume, d_ndegf, &
                                          pmass, half_dt, ekin, bmoment)

    type(s_ensemble),        intent(in)    :: ensemble
    real(wip),               intent(in)    :: press(:)
    real(wip),               intent(in)    :: pressxyz
    real(wip),               intent(in)    :: pressxy
    real(wip),               intent(in)    :: press0
    real(wip),               intent(in)    :: volume
    real(wip),               intent(in)    :: d_ndegf
    real(wip),               intent(in)    :: pmass
    real(wip),               intent(in)    :: half_dt
    real(dp),                intent(in)    :: ekin
    real(wip),               intent(inout) :: bmoment(:)

    real(wip)                :: ekin_real

    ekin_real = real(ekin,wip)

    if (ensemble%isotropy == IsotropyISO) then
      bmoment(1) = bmoment(1) + half_dt*(3.0_wip*volume*(pressxyz - press0) &
                              + 6.0_wip*ekin_real/d_ndegf)/pmass
      bmoment(2) = bmoment(1)
      bmoment(3) = bmoment(1)
    else if (ensemble%isotropy == IsotropySEMI_ISO) then
      bmoment(1) = bmoment(1) + half_dt*(volume*(pressxy - press0)   &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
      bmoment(2) = bmoment(2) + half_dt*(volume*(pressxy - press0)   &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
      bmoment(3) = bmoment(3) + half_dt*(volume*(press(3) - press0)  &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
    else if (ensemble%isotropy == IsotropyANISO) then
      bmoment(1) = bmoment(1) + half_dt*(volume*(press(1) - press0)  &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
      bmoment(2) = bmoment(2) + half_dt*(volume*(press(2) - press0)  &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
      bmoment(3) = bmoment(3) + half_dt*(volume*(press(3) - press0)  &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
    else if (ensemble%isotropy == IsotropyXY_Fixed) then
      bmoment(1) = 0.0_wip
      bmoment(2) = 0.0_wip
      bmoment(3) = bmoment(3) + half_dt*(volume*(press(3) - press0) &
                              + 2.0_wip*ekin_real/d_ndegf)/pmass
    end if

  end subroutine gpu_core_update_barostat_mtk

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_collect
  !> @brief        fold one native force evaluation into s_dynvars
  !
  !  The native core reports finer energy slots than GENESIS's totals: the
  !  1-4 terms and the excluded-pair correction land in the electrostatic and
  !  van der Waals totals the CPU path puts them in, the Ewald self-energy
  !  with them. With MPI the bonded slots reach the ranks' sum exactly: this
  !  rank's doubles leave them out (zero) and their fixed-point words ride in
  !  compute_dynvars' own allreduce (reduce_property's armed extra words),
  !  folded back by gpu_core_exact_fold; the NPT virial allreduce carries the
  !  virial slots the same way, so the global bonded energies and virial are
  !  the same bits for any partition of the terms.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_collect(result, dynvars, enefunc, boundary)

    type(s_gc_step_result),  intent(in)    :: result
    type(s_dynvars), target, intent(inout) :: dynvars
    type(s_enefunc),         intent(in)    :: enefunc
    type(s_boundary),        intent(in)    :: boundary

    integer                  :: k
    real(wip)                :: volume

    exact_words(1:GcExactNWord) = result%bonded_exact(1:GcExactNWord)

#ifdef HAVE_MPI_GENESIS
    dynvars%energy%bond         = 0.0_dp
    dynvars%energy%angle        = 0.0_dp
    dynvars%energy%urey_bradley = 0.0_dp
    dynvars%energy%dihedral     = 0.0_dp
    dynvars%energy%improper     = 0.0_dp
    dynvars%energy%cmap         = 0.0_dp
    dynvars%energy%electrostatic = result%energy(GcEneElecReal)  &
                                 + result%energy(GcEneElecRecip) &
                                 + result%energy(GcEneElecSelf)
    dynvars%energy%van_der_waals = result%energy(GcEneVdwReal)
    dynvars%energy%restraint_position = 0.0_dp
    reduce_extra(1:GcExactNWord) = result%bonded_exact(1:GcExactNWord)
    reduce_extra_n     = GcExactNWord
    reduce_extra_fold  = c_funloc(gpu_core_exact_fold)
    reduce_extra_vfold = c_funloc(gpu_core_exact_vfold)
#else
    dynvars%energy%bond         = result%energy(GcEneBond)
    dynvars%energy%angle        = result%energy(GcEneAngle)
    dynvars%energy%urey_bradley = result%energy(GcEneUrey)
    dynvars%energy%dihedral     = result%energy(GcEneDihedral)
    dynvars%energy%improper     = result%energy(GcEneImproper)
    dynvars%energy%cmap         = result%energy(GcEneCmap)
    dynvars%energy%electrostatic = result%energy(GcEneElecReal)  &
                                 + result%energy(GcEneElec14)    &
                                 + result%energy(GcEneElecRecip) &
                                 + result%energy(GcEneElecSelf)  &
                                 + result%energy(GcEneElecCorr)
    dynvars%energy%van_der_waals = result%energy(GcEneVdwReal)   &
                                 + result%energy(GcEneVdw14)
    dynvars%energy%restraint_position = result%energy(GcEnePosres)
#endif

    ! compute_energy_charmm's total, with the reciprocal term already inside
    ! electrostatic; stock temperature_remd judges an exchange by this value,
    ! so it must be this step's. electric_field is the host value (a run
    ! that has one is declined).
    dynvars%energy%total = dynvars%energy%bond               &
                         + dynvars%energy%angle              &
                         + dynvars%energy%urey_bradley       &
                         + dynvars%energy%dihedral           &
                         + dynvars%energy%cmap               &
                         + dynvars%energy%improper           &
                         + dynvars%energy%electrostatic      &
                         + dynvars%energy%van_der_waals      &
                         + dynvars%energy%electric_field     &
                         + dynvars%energy%restraint_position

    ! The dispersion correction as compute_energy evaluates it, from the
    ! reference box's volume, which a barostat step changes; the energy is
    ! added by compute_dynvars, the virial here on the rank stock adds it on.
    if (enefunc%dispersion_corr /= Disp_corr_NONE) then
      volume = boundary%box_size_x_ref * boundary%box_size_y_ref &
             * boundary%box_size_z_ref
      dynvars%energy%disp_corr_energy = enefunc%dispersion_energy / volume
      if (enefunc%dispersion_corr == Disp_corr_EPress) &
        dynvars%energy%disp_corr_virial = enefunc%dispersion_virial / volume
    end if
    do k = 1, 3
#ifdef HAVE_MPI_GENESIS
      dynvars%virial(k,k) = result%virial_nb(k)
      dynvars%virial_extern(k,k) = 0.0_dp
#else
      dynvars%virial(k,k) = result%virial(k)
      dynvars%virial_extern(k,k) = result%virial_ext(k)
#endif
      if (replica_main_rank .or. main_rank) &
        dynvars%virial(k,k) = dynvars%virial(k,k) &
                            + dynvars%energy%disp_corr_virial
    end do

  end subroutine gpu_core_collect

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_exact_fold
  !> @brief        fold the bonded words, summed over the ranks, into
  !!               reduce_property's values (its layout: 1 bond, 3 angle,
  !!               4 urey_bradley, 5 dihedral, 6 improper, 7 cmap,
  !!               8 electrostatic, 9 van_der_waals, 10 electric_field,
  !!               11 restraint_position, 12 total, 13:15 virial,
  !!               16:18 virial_ext); the total is formed again as
  !!               gpu_core_collect forms it
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_exact_fold(nextra, extra, reduced) &
             bind(c, name='gpu_core_exact_fold_f')

    integer(c_int),   value,  intent(in)    :: nextra
    real(c_double),           intent(in)    :: extra(nextra)
    real(c_double),           intent(inout) :: reduced(20)

    real(c_double)           :: v(GcExactNSlot)
    integer                  :: k

    if (nextra /= GcExactNWord) &
      call gpu_core_abort('Gpu_Core_Collect> exact bonded words')
    call gpu_core_exact_decode(extra, int(GcExactNSlot, c_int32_t), v)

    reduced(1)  = v(GcExactBond)
    reduced(3)  = v(GcExactAngle)
    reduced(4)  = v(GcExactUrey)
    reduced(5)  = v(GcExactDihe)
    reduced(6)  = v(GcExactImpr)
    reduced(7)  = v(GcExactCmap)
    reduced(8)  = reduced(8) + v(GcExactElec14) + v(GcExactElecCor)
    reduced(9)  = reduced(9) + v(GcExactVdw14)
    reduced(11) = v(GcExactPosres)
    reduced(12) = reduced(1) + reduced(3) + reduced(4) + reduced(5) &
                + reduced(7) + reduced(6) + reduced(8) + reduced(9) &
                + reduced(10) + reduced(11)
    do k = 1, 3
      reduced(12+k) = reduced(12+k) - v(GcExactVirX+k-1)
      reduced(15+k) = reduced(15+k) + v(GcExactPVirX+k-1)
    end do

  end subroutine gpu_core_exact_fold

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_exact_vfold
  !> @brief        fold the summed bonded virial words into a virial diagonal,
  !!               for stock's barostat reduction (reduce_virial_sum) in the
  !!               extra VV1 the native loop leaves to stock
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_exact_vfold(nextra, extra, diag) &
             bind(c, name='gpu_core_exact_vfold_f')

    integer(c_int),   value,  intent(in)    :: nextra
    real(c_double),           intent(in)    :: extra(nextra)
    real(c_double),           intent(inout) :: diag(3)

    real(c_double)           :: v(3)
    integer                  :: i0

    if (nextra /= GcExactNWord) &
      call gpu_core_abort('Gpu_Core_Collect> exact bonded words')
    i0 = 3*(GcExactVirX-1)
    call gpu_core_exact_decode(extra(i0+1:i0+9), 3_c_int32_t, v)
    diag(1:3) = diag(1:3) - v(1:3)

  end subroutine gpu_core_exact_vfold

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_npt_virial_sum
  !> @brief        the barostat's global virial diagonal: this rank's in,
  !!               the global one out, the bonded part riding the allreduce
  !!               as exact words (gpu_core_collect)
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_npt_virial_sum(virial_sum)

    real(dp),                intent(inout) :: virial_sum(3)

    real(dp)                 :: buf(12)
    real(c_double)           :: v(3)
    integer                  :: i0

#ifdef HAVE_MPI_GENESIS
    i0 = 3*(GcExactVirX-1)
    buf(1:3)  = virial_sum(1:3)
    buf(4:12) = exact_words(i0+1:i0+9)
    call mpi_allreduce(mpi_in_place, buf, 12, mpi_real8, mpi_sum, &
                       mpi_comm_country, ierror)
    call gpu_core_exact_decode(buf(4:12), 3_c_int32_t, v)
    virial_sum(1:3) = buf(1:3) - v(1:3)
#endif

  end subroutine gpu_core_npt_virial_sum

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Function      gpu_core_graph_kind
  !> @brief        whether step i is plain -- nothing between step_begin and
  !!               the VV2 constraint reads the device or queues host data
  !!               mid-step: no energy, virial, output, list build, guard
  !!               flush, COM removal, cons_nvt kinetic, flat-NVE reference
  !!               kinetic, r-RESPA or barostat -- and of which graph kind
  !!               (gpu_core_graph_begin)
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  function gpu_core_graph_kind(i, istart, iend, dynamics, ensemble,       &
                               cons_nvt, flat_nve, list_step)
    integer,                 intent(in) :: i, istart, iend
    type(s_dynamics),        intent(in) :: dynamics
    type(s_ensemble),        intent(in) :: ensemble
    logical,                 intent(in) :: cons_nvt, flat_nve
    logical, optional,       intent(in) :: list_step
    integer(c_int32_t)                  :: gpu_core_graph_kind

    logical :: plain, tick

    tick  = (ensemble%ensemble == EnsembleNVT) .and.                      &
            (mod(i-1, dynamics%thermo_period) == 0) .and. (i > 1)
    plain = mod(i-1, dynamics%eneout_period) /= 0 .and. .not. cons_nvt .and.&
            i > istart .and. i < iend .and.                               &
            mod(i, dynamics%eneout_period) /= 0 .and.                     &
            mod(i-istart+1, GcGuardRing) /= 0
    if (present(list_step)) then
      plain = plain .and. .not. list_step
    else if (dynamics%nbupdate_period > 0 .and. i > 1) then
      plain = plain .and. mod(i-1, dynamics%nbupdate_period) /= 0
    end if
    if (dynamics%stoptr_period > 0) plain = plain .and.                   &
      mod(i-1, dynamics%stoptr_period) /= 0
    if (dynamics%crdout_period > 0) plain = plain .and.                   &
      mod(i, dynamics%crdout_period) /= 0
    if (dynamics%velout_period > 0) plain = plain .and.                   &
      mod(i, dynamics%velout_period) /= 0
    if (dynamics%rstout_period > 0) plain = plain .and.                   &
      mod(i, dynamics%rstout_period) /= 0
    if (flat_nve) plain = plain .and. mod(i-2, dynamics%eneout_period) /= 0
    plain = plain .and. dynamics%gpu_step_graph .and.                     &
            dynamics%integrator /= IntegratorVRES .and.                   &
            (ensemble%ensemble == EnsembleNVE .or.                        &
             ensemble%ensemble == EnsembleNVT)
    gpu_core_graph_kind = GcGraphNone
    if (plain) gpu_core_graph_kind = merge(GcGraphTick, GcGraphPlain, tick)

  end function gpu_core_graph_kind

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_vverlet
  !> @brief        the native velocity-Verlet loop (VVER and VRES)
  !! @param[out]   ok        : .false. when the core refused and the caller
  !!                           must run the stock loop
  !! @param[out]   tail_done : .true. when the loop's tail (extra VV1,
  !!                           reference restore, energy output) is done
  !
  !  The structure follows vverlet_dynamics (vverlet_respa_dynamics): the
  !  same scheduled positions for output, energy output, centre-of-mass
  !  removal and the pair-list update, and the same order of VV1,
  !  constraints, force and VV2, none of it moving particle data.
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_vverlet(output, domain, enefunc, dynvars,           &
                              dynamics, pairlist, boundary, constraints,  &
                              ensemble, remd, ok, tail_done)

    type(s_output),           intent(inout) :: output
    type(s_domain),   target, intent(inout) :: domain
    type(s_enefunc),          intent(inout) :: enefunc
    type(s_dynvars),  target, intent(inout) :: dynvars
    type(s_dynamics),         intent(inout) :: dynamics
    type(s_pairlist),         intent(inout) :: pairlist
    type(s_boundary),         intent(inout) :: boundary
    type(s_constraints),      intent(inout) :: constraints
    type(s_ensemble),         intent(inout) :: ensemble
    type(s_remd),             intent(inout) :: remd
    logical,                  intent(out)   :: ok
    logical, optional,        intent(out)   :: tail_done

    !  The context is a local of this call, created from the frozen state at
    !  entry and destroyed at exit; in a temperature-REMD run `core` is the
    !  module's resident context instead.
    type(s_gpu_core), target       :: local_core
    type(s_gpu_core), pointer      :: core
    logical                  :: resident, resumed
    type(s_gc_step_result), target :: result
    character(128)           :: reason
    integer                  :: i, istart, iend, status, k
    integer                  :: ens_kind, thermo, group_tp
    integer                  :: age
    integer(iintegers)       :: num_degree
    integer(c_int64_t)       :: nfail
    real(dp)                 :: dt, half_dt, dt_therm
    real(c_double)           :: guard_ring(GcGuardRing)
    integer(c_int32_t)       :: guard_n, defer_vv2, want_virial, tick_next
    integer(c_int32_t)       :: routes(3)
    integer(c_int64_t)       :: guard_kept_over
    real(c_double)           :: kin_half3(3), kin_full3(3), viri3(3)
    real(c_double), target   :: rmsg_local, ekin_sum_local
    real(dp)                 :: ekin_half, ekin_full
    real(wip)                :: scale_vel, scale_vel2, rr
    logical                  :: calc_thermostat, want_energy, boundary_due
    logical                  :: flat_nve, cons_nvt, scheduled
    logical                  :: guard_on, early
    integer                  :: last_build
    real(c_double)           :: guard_val(2), guard_bound
    integer(c_int32_t)       :: graph_kind
    logical                  :: device_thermo
    integer                  :: draws_left
    real(c_double)           :: thermo_draw(2, GcThermoDraws)
    logical                  :: respa
    integer                  :: multistep, jstep
    real(c_double)           :: half_dt_long
    logical                  :: npt
    real(c_double)           :: npt_scale(3), npt_bmoment(3)

    if (present(tail_done)) tail_done = .false.
    resident = .false.
    if (allocated(remd%types))                                            &
      resident = remd%dimension >= 1 .and. remd%total_nreplicas >= 2 .and.&
                 all(remd%types(1:remd%dimension) == RemdTemperature) .and.&
                 ensemble%ensemble == EnsembleNVT
    resumed  = .false.
    core => local_core
    if (resident) core => remd_core

    ! Admission: the local answer, then one collective decision per
    ! simulation communicator.
    ok = gpu_core_admit(dynamics, ensemble, constraints, domain, enefunc, &
                        remd, reason)
    if (nproc_country > 1) then
      k = merge(GcOk, GcErrUnsupport, ok)
      if (.not. ok .and. .not. main_rank) &
        write(MsgOut,'(A,I0,A)') 'Setup_GPU_Core> rank ', my_country_rank, &
          ' declines: ' // trim(reason)
      call gpu_core_vote(k, nproc_country)
      if (ok .and. k /= GcOk) then
        ok = .false.
        reason = 'a peer rank declined native eligibility'
      end if
    end if
    if (.not. ok) then
      if (main_rank) write(MsgOut,'(A)')                                  &
        'Setup_GPU_Core> requested=YES effective=cpu reason=' // trim(reason)
      call gpu_core_decline(enefunc, reason)
      return
    end if

    dt      = dynamics%timestep / AKMA_PS
    half_dt = 0.5_dp * dt
    respa   = dynamics%integrator == IntegratorVRES
    multistep = 1
    if (respa) multistep = dynamics%elec_long_period
    half_dt_long = real(0.5_dp * dt * real(multistep, dp), c_double)
    npt = ensemble%ensemble /= EnsembleNVE .and. &
          ensemble%ensemble /= EnsembleNVT

    ! A resident temperature-REMD context resumes: state, lists and the last
    ! force are on the device, and the exchange has already rescaled its
    ! velocities there; stock re-enters vverlet_dynamics the same way.
    if (resident .and. remd_live) then
      resumed = .true.
      if (main_rank) write(MsgOut,'(A,I0)')                               &
        'Setup_GPU_Core> requested=YES effective=native resident=REMD '// &
        'resumed_at_step=', dynamics%istart_step - 1
    end if

    if (.not. resumed) then
    call timer(TimerNativeSetup, TimerOn)

    ! [DYNAMICS] gpu_route_*, as gcx_route_mode (MPI 0, THREAD 1); the
    ! transport reads them in its start-up probe
    routes(1) = int(dynamics%gpu_route_mesh  - 1, c_int32_t)
    routes(2) = int(dynamics%gpu_route_coord - 1, c_int32_t)
    routes(3) = int(dynamics%gpu_route_force - 1, c_int32_t)
    k = gcx_route_select(routes)

    call gpu_core_setup(core, domain, enefunc, constraints, boundary,    &
                        dynamics%nbupdate_period, my_country_rank,       &
                        nproc_country, my_country_no, status)
    if (status == GcOk .and. .not. core%active) status = GcErrState
    call setup_vote('a peer rank declined the native core setup', .false.)
    if (status /= GcOk) return

    ! The real-space zero bits are the step's input: gpu_step.cu refuses
    ! without them, so a communicator that reaches here must admit them or
    ! fall back whole.
    call gpu_core_admit_real_mask(core, domain, enefunc, status)
    call setup_vote('a peer rank declined the real-space mask', .false.)
    if (status /= GcOk) return

    call gpu_core_attach_posres(core, domain, enefunc, status)
    call setup_vote('a peer rank declined the positional restraints', .false.)
    if (status /= GcOk) return

    if (main_rank) then
      if (enefunc%nonbond_precision == NonbondPrecisionMixed) then
        write(MsgOut,'(A)')                                               &
          'Setup_GPU_Core> requested=YES effective=native precision=MIXED '//&
          'reason=eligible'
      else
        write(MsgOut,'(A)')                                               &
          'Setup_GPU_Core> requested=YES effective=native precision=FP64 '//&
          'reason=eligible'
      end if
      write(MsgOut,'(A,I6,A,F10.5,A,F10.5,A,F10.5)')                      &
        'Setup_GPU_Core> cadence_max=', dynamics%nbupdate_period,         &
        ' list_radius=', core%list_radius,                                &
        ' support_radius=', core%support_radius,                          &
        ' half_skin=', core%half_skin
      write(MsgOut,'(A,I12,A,I10)')                                       &
        'Setup_GPU_Core> local_owned=', core%num_owned,                   &
        ' halo=', core%num_halo
    end if

    ens_kind = GcEnsembleNVE
    thermo   = GcThermoNone
    if (ensemble%ensemble /= EnsembleNVE) then
      ens_kind = merge(GcEnsembleNPT, GcEnsembleNVT, npt)
      if (ensemble%tpcontrol == TpcontrolBerendsen) thermo = GcThermoBerend
      if (ensemble%tpcontrol == TpcontrolBussi)     thermo = GcThermoBussi
      if (ensemble%tpcontrol == TpcontrolNHC)       thermo = GcThermoNHC
    end if
    group_tp = merge(1, 0, ensemble%group_tp)

    call gpu_core_attach_step(core, domain, enefunc, constraints,        &
                              ens_kind, thermo, group_tp, dt,            &
                              dynamics%nbupdate_period,                  &
                              dynamics%thermo_period, status)
    ! the acceptance above is revoked, out loud: the step plan is the last
    ! thing that can decline
    call setup_vote('a peer rank declined the native step plan', .true.)
    if (status /= GcOk) return

    status = int(gpu_core_rebuild(core%ctx, 0_c_int32_t))
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Vverlet> the first rebuild failed: ' // &
                     trim(gc_status_text(status)))
    if (respa) then
      status = int(gpu_core_force_respa(core%ctx, 1_c_int32_t, 1_c_int32_t, &
                                        1_c_int32_t, c_loc(result)))
    else
      status = int(gpu_core_force(core%ctx, 1_c_int32_t, 1_c_int32_t, &
                                  c_loc(result)))
    end if
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Vverlet> the first force evaluation '// &
                     'failed: ' // trim(gc_status_text(status)))
    call gpu_core_collect(result, dynvars, enefunc, boundary)

    ! From here the stock loop does not run in this call, nor, unless a
    ! non-resident REMD cycle re-enters, in any later one: free the stock
    ! pair-list and nonbond device buffers the step-0 force left behind.
    if (resident .or. .not. allocated(remd%types)) &
      call gpu_release_stock_buffers()

    call timer(TimerNativeSetup, TimerOff)
    if (resident) remd_live = .true.
    end if

    istart     = dynamics%istart_step
    iend       = dynamics%iend_step
    dt_therm   = dt * real(dynamics%thermo_period, dp)
    num_degree = domain%num_deg_freedom
    if (ensemble%group_tp) num_degree = domain%num_group_freedom
    age = 0
    if (resumed) age = remd_age
    guard_on = dynamics%gpu_list_guard .and. .not. respa .and. .not. npt &
               .and. dynamics%nbupdate_period > 0
    core%list_guard = guard_on
    if (dynamics%gpu_list_guard .and. .not. guard_on .and. main_rank)     &
      write(MsgOut,'(A)') 'Setup_GPU_Core> gpu_list_guard = YES is not '// &
        'used with r-RESPA or NPT: the list follows nbupdate_period'
    last_build = istart - 1 - age
    flat_nve = (ensemble%ensemble == EnsembleNVE) .and. .not. ensemble%group_tp
    cons_nvt = (ensemble%ensemble == EnsembleNVT) .and.                   &
               constraints%rigid_bond .and. .not. ensemble%group_tp
    device_thermo = (ensemble%ensemble == EnsembleNVT) .and. .not. cons_nvt
    draws_left    = 0
    if (device_thermo) call thermostat_start

    do i = istart, iend

      ! sp_md_respa.fpp advances time and step once per outer step and
      ! writes trajectories and restarts only there
      jstep = mod(i - istart, multistep) + 1
      if (jstep == 1) then
        dynvars%time = dynamics%timestep * real(i-1,dp)
        dynvars%step = i-1
      end if

      want_energy  = (mod(i-1, dynamics%eneout_period) == 0)
      boundary_due = output_due(i-1)

      if (i > istart .and. boundary_due .and. jstep == 1) then
        call timer(TimerNativeOutput, TimerOn)
        call gpu_core_pull(core, domain, status)
        if (status /= GcOk) &
          call gpu_core_abort('Gpu_Core_Vverlet> output boundary pull failed')
        call output_md(output, dynamics, boundary, pairlist, ensemble,   &
                       constraints, dynvars, domain, enefunc, remd)
        call timer(TimerNativeOutput, TimerOff)
      end if

      call timer(TimerIntegrator, TimerOn)
      call timer(TimerNativeStep, TimerOn)

      ! Queued before the step's launches (and after any launches still
      ! pending), so a tick's step stays plain; the host RNG is not drawn
      ! from between here and the tick, so the stock order is unchanged.
      if (ensemble%ensemble == EnsembleNVT .and. .not. cons_nvt .and.     &
          ensemble%tpcontrol == TpcontrolBussi .and. i > 1 .and.          &
          mod(i-1, dynamics%thermo_period) == 0 .and. draws_left == 0)    &
        call bussi_draws(i)

      ! The host knows the guard of step i-2 without synchronising the
      ! device (the device still has step i-1 queued): d_max(i) <=
      ! d_max(i-2) + the two moves since, each taken as twice step i-2's
      ! largest move.  A list built at step i-2 or i-1 starts from zero.
      ! The device counts the kept steps whose own d_max reached the
      ! buffer (GPU_Core_Summary).
      early = .false.
      if (guard_on) then
        scheduled = i > 1 .and. age + 1 >= dynamics%nbupdate_period
        if (.not. scheduled .and. i >= istart + 2) then
          call gc_check(gpu_core_list_guard_read(core%ctx, 2_c_int32_t,   &
                        guard_val), 'Gpu_Core_Vverlet> list guard read')
#ifdef HAVE_MPI_GENESIS
          if (nproc_country > 1) &
            call mpi_allreduce(mpi_in_place, guard_val, 2, mpi_real8,     &
                               mpi_max, mpi_comm_country, ierror)
#endif
          if (last_build >= i - 2) then
            guard_bound = real(i - last_build, c_double) * 2.0_dp*guard_val(2)
          else
            guard_bound = guard_val(1) + 4.0_dp*guard_val(2)
          end if
          early = guard_bound >= 2.0_dp*core%half_skin
        end if
        scheduled = scheduled .or. early
        graph_kind = gpu_core_graph_kind(i, istart, iend, dynamics,        &
                                         ensemble, cons_nvt, flat_nve,     &
                                         scheduled)
      else
        graph_kind = gpu_core_graph_kind(i, istart, iend, dynamics,        &
                                         ensemble, cons_nvt, flat_nve)
      end if
      call gc_check(gpu_core_graph_begin(core%ctx, graph_kind),           &
                    'Gpu_Core_Vverlet> graph begin')

      call gc_check(gpu_core_step_begin(core%ctx, int(i, c_int64_t),     &
                    merge(1_c_int32_t, 0_c_int32_t, want_energy)),       &
                    'Gpu_Core_Vverlet> step_begin')

      ! nve_vv1 (sp_md_vverlet.fpp) copies the velocity into velocity_ref,
      ! forms velocity_half = half_dt/m * force, reduces both, and only
      ! then kicks; step_begin has just made the same two preparations on
      ! the device, so the pair is sampled here, at the stock phase.
      if (flat_nve .and. want_energy) then
        ! nve_vv1_nogroup: the full term is the pre-kick velocity; at step
        ! 1 it also seeds ekin_ref; the half term is sampled after RATTLE
        call kinetic(GcKinFlatVel, kin_full3, ekin_full,                  &
                     'Gpu_Core_Vverlet> kinetic full')
        dynvars%kin_full(1:3) = kin_full3(1:3)
        dynvars%ekin_full     = ekin_full
        if (i == 1) dynvars%ekin_ref = ekin_full
      else if (ensemble%ensemble == EnsembleNVE .and. want_energy) then
        call kinetic(merge(GcKinGroupVelHalf, GcKinFlatVelHalf,           &
                           ensemble%group_tp), kin_half3, ekin_half,      &
                     'Gpu_Core_Vverlet> kinetic half')
        call kinetic(merge(GcKinGroupVel, GcKinFlatVel, ensemble%group_tp),&
                     kin_full3, ekin_full, 'Gpu_Core_Vverlet> kinetic full')
        call set_kinetic()
        dynvars%ekin          = ekin_full + 2.0_dp*ekin_half/3.0_dp
        dynvars%kin(1:3)      = kin_full3(1:3) + kin_half3(1:3)
      end if

      if (npt) then
        call npt_vv1(i)
      else if (cons_nvt) then
        call cons_nvt_vv1(i)
      else

      scale_vel       = 1.0_wip
      calc_thermostat = (ensemble%ensemble == EnsembleNVT) .and.          &
                        (mod(i-1, dynamics%thermo_period) == 0) .and.     &
                        (i > 1)
      if (respa) calc_thermostat = calc_thermostat .and. (i > istart)

      if (calc_thermostat) then
        ! Stock's tick (vel_scale_* on the global kinetic energy, then a
        ! broadcast of rank 0's scale) runs on the device: every rank forms
        ! the same scale from the same fixed-point sums and the same draws,
        ! and the VV1 reads it there.  The Bussi draws are taken ahead.
        if (respa) then
          call gc_check(gpu_core_respa_half(core%ctx),                    &
                        'Gpu_Core_Vverlet> NVT half velocity')
        else
          call gc_check(gpu_core_nvt_half(core%ctx),                      &
                        'Gpu_Core_Vverlet> NVT half velocity')
        end if
        if (ensemble%tpcontrol == TpcontrolBussi) draws_left = draws_left - 1
        call gc_check(gpu_core_thermostat(core%ctx,                       &
                      merge(GcKinGroupVelHalf, GcKinFlatVelHalf,          &
                            ensemble%group_tp),                           &
                      merge(GcKinGroupVelRef, GcKinFlatVelRef,            &
                            ensemble%group_tp)),                          &
                      'Gpu_Core_Vverlet> thermostat')
        if (respa) call thermostat_publish(scale_vel)
      end if

      if (ensemble%ensemble == EnsembleNVT .and. want_energy) then
        ! stock's pre-constraint VV1 observables; on a non-tick stock keeps
        ! the full/half values of the preceding tick with unit scaling
        call thermostat_publish(scale_vel)
        if (.not. calc_thermostat) scale_vel = 1.0_wip
        call nvt_observables()
      end if

      if (respa) then
        call gc_check(gpu_core_respa_vv1(core%ctx,                        &
                      real(scale_vel, c_double),                          &
                      merge(half_dt_long, 0.0_c_double, jstep == 1)),     &
                      'Gpu_Core_Vverlet> vv1')
      else
        call gc_check(gpu_core_vv1(core%ctx, 1.0_c_double,                &
                      merge(GcVv1DeviceScale, GcVv1Kick, calc_thermostat)),&
                      'Gpu_Core_Vverlet> vv1')
      end if

      ! Only an output step reads VV1's constraint virial; elsewhere the
      ! failure count is read at this step's force join.
      call constrain(GcConstrainVV1 + merge(0_c_int32_t, GcConstrainDefer, &
                     want_energy), 'Gpu_Core_Vverlet> a constraint group '//&
                     'did not converge; see GPU_Core_Constraint')
      ! stock r-RESPA keeps the constraint virial out of the virial
      do k = 1, 3
        dynvars%virial_const(k,k) = viri3(k)
        if (.not. respa) dynvars%virial(k,k) = dynvars%virial(k,k) + viri3(k)
      end do

      ! nve_vv1_nogroup: on an output step the post-RATTLE velocity is the
      ! half term and becomes the next reference; on the step after one it
      ! refreshes the reference; at step 1 the constraint virial is added a
      ! second time, as stock does.
      if (flat_nve) then
        if (want_energy) then
          call kinetic(GcKinFlatVel, kin_half3, ekin_half,                &
                       'Gpu_Core_Vverlet> kinetic half')
          if (i == 1) then
            dynvars%kin_ref(1:3) = kin_half3(1:3)
            do k = 1, 3
              dynvars%virial(k,k) = dynvars%virial(k,k) + viri3(k)
            end do
          end if
          dynvars%kin_half(1:3) = kin_half3(1:3)
          dynvars%ekin_half     = ekin_half
          dynvars%ekin     = (dynvars%ekin_full + dynvars%ekin_half       &
                              + dynvars%ekin_ref) / 3.0_dp
          dynvars%kin(1:3) = 0.5_dp * (dynvars%kin_ref(1:3)               &
                                       + dynvars%kin_half(1:3))
          dynvars%ekin_ref     = dynvars%ekin_half
          dynvars%kin_ref(1:3) = dynvars%kin_half(1:3)
        else if (mod(i-2, dynamics%eneout_period) == 0) then
          call kinetic(GcKinFlatVel, kin_half3, ekin_half,                &
                       'Gpu_Core_Vverlet> kinetic ref')
          dynvars%kin_ref(1:3) = kin_half3(1:3)
          dynvars%ekin_ref     = ekin_half
        end if
      end if

      end if   ! npt, cons_nvt

      call timer(TimerNativeStep, TimerOff)
      call timer(TimerIntegrator, TimerOff)

      if (want_energy .and. i > istart) then
        call timer(TimerNativeEnergy, TimerOn)
        ! mtk_barostat_vv1 adds the group virial on barostat steps only,
        ! which npt_vv1 has done.
        if (ensemble%group_tp .and. .not. npt) &
          call add_group_virial('Gpu_Core_Vverlet> group virial')
        call energy_output('Gpu_Core_Vverlet> scalar observable reduction')
        call timer(TimerNativeEnergy, TimerOff)
      end if

      if (dynamics%stoptr_period > 0) then
        if (mod(i-1, dynamics%stoptr_period) == 0)                        &
          call gc_check(gpu_core_com_removal(core%ctx,                    &
                        merge(1_c_int32_t, 0_c_int32_t,                   &
                              dynamics%stop_com_translation),             &
                        merge(1_c_int32_t, 0_c_int32_t,                   &
                              dynamics%stop_com_rotation)),               &
                        'Gpu_Core_Vverlet> com')
      end if

      call timer(TimerPairList, TimerOn)
      age = age + 1
      ! The scheduled rebuild is stock's: domain_interaction_update(i-1)
      ! for i > 1 migrates and rebuilds when mod(i-1, nbupdate_period) is
      ! zero, so the cell each atom is in -- which decides the within-cell
      ! pairs of stock's CUTOFF force-only clamp -- changes on the same
      ! steps as stock's.
      if (.not. guard_on) then
        scheduled = dynamics%nbupdate_period > 0 .and. i > 1
        if (scheduled) scheduled = mod(i-1, dynamics%nbupdate_period) == 0
      end if
      ! r-RESPA: domain_interaction_update(istep) on the last inner step of
      ! every outer step but the first, before the full force
      if (respa) scheduled = dynamics%nbupdate_period > 0 .and.          &
                             jstep == multistep .and.                     &
                             i - multistep + 1 > istart .and.             &
                             mod(i, dynamics%nbupdate_period) == 0
      ! Stock rebuilds on its schedule only, and so does the native step
      ! unless [DYNAMICS] gpu_list_guard = YES (decided at the top of the
      ! step).  The displacement guard is queued on the device and reduced
      ! in one MPI_MAX over the queued steps when the queue is full or at
      ! the last step.  Stock's own list routinely exceeds the half skin
      ! between scheduled rebuilds, so on stock's schedule the largest
      ! displacement and the count of such steps are reported, not refused.
      call gc_check(gpu_core_list_guard_defer(core%ctx,                   &
                    merge(1_c_int32_t, 0_c_int32_t,                       &
                          guard_on .and. .not. scheduled)),               &
                    'Gpu_Core_Vverlet> list guard')
      guard_n = GcGuardRing
      if (i == iend .or. mod(i-istart+1, GcGuardRing) == 0) then
        call gc_check(gpu_core_list_guard_flush(core%ctx, guard_ring,     &
                      guard_n, guard_kept_over), 'Gpu_Core_Vverlet> list guard')
#ifdef HAVE_MPI_GENESIS
        if (guard_n > 0) &
          call mpi_allreduce(mpi_in_place, guard_ring, int(guard_n),       &
                             mpi_real8, mpi_max, mpi_comm_country, ierror)
#endif
        do k = 1, int(guard_n)
          core%guard_dmax = max(core%guard_dmax, guard_ring(k))
          if (.not. guard_on .and.                                        &
              2.0_dp*guard_ring(k) >= 2.0_dp*core%half_skin)         &
            core%guard_over = core%guard_over + 1
        end do
        if (guard_on) then
#ifdef HAVE_MPI_GENESIS
          call mpi_allreduce(mpi_in_place, guard_kept_over, 1,             &
                             mpi_integer8, mpi_max, mpi_comm_country, ierror)
#endif
          core%guard_over = guard_kept_over
        end if
      end if
      if (scheduled) then
        call timer(TimerNativeRebuild, TimerOn)
        status = int(gpu_core_rebuild(core%ctx,                           &
                     merge(1_c_int32_t, 0_c_int32_t, early)))
        age = 0
        last_build = i
        if (status /= GcOk) &
          call gpu_core_abort('Gpu_Core_Vverlet> rebuild: ' // &
                         trim(gc_status_text(status)))
        call timer(TimerNativeRebuild, TimerOff)
      end if
      call timer(TimerPairList, TimerOff)

      ! The force virial and VV2's constraint virial are read by the next
      ! output step only (and by the tail after the last step).  VV2's
      ! failure count waits for the next force join unless an output
      ! boundary pulls the state first.  A barostat step's force is
      ! compute_energy's npt1 = mod(i, baro_period) == 0: the next VV1's
      ! pressure reads its virial.
      want_virial = merge(1_c_int32_t, 0_c_int32_t,                       &
                          mod(i, dynamics%eneout_period) == 0 .or.        &
                          i == iend .or.                                  &
                          (npt .and. mod(i, dynamics%baro_period) == 0))
      defer_vv2 = merge(GcConstrainDefer, 0_c_int32_t, want_virial == 0)
      if (output_due(i)) defer_vv2 = 0

      call timer(TimerEnergy, TimerOn)
      if (respa) then
        status = int(gpu_core_force_respa(core%ctx,                       &
                     merge(1_c_int32_t, 0_c_int32_t,                      &
                           mod(i, dynamics%eneout_period) == 0),          &
                     want_virial,                                         &
                     merge(1_c_int32_t, 0_c_int32_t, jstep == multistep), &
                     c_loc(result)))
      else
        status = int(gpu_core_force(core%ctx,                             &
                     merge(1_c_int32_t, 0_c_int32_t,                      &
                           mod(i, dynamics%eneout_period) == 0),          &
                     want_virial, c_loc(result)))
      end if
      if (status /= GcOk) &
        call gpu_core_abort('Gpu_Core_Vverlet> force: ' // &
                       trim(gc_status_text(status)))
      call gpu_core_collect(result, dynvars, enefunc, boundary)
      call timer(TimerEnergy, TimerOff)

      ! The restart writer serializes the stocked RNG, not the live state.
      ! Match the stock loop's post-force, pre-VV2 publication point so a
      ! native Bussi draw is present in the next output boundary's restart
      ! (sp_md_respa.fpp has no such publication point).
      if (.not. respa) call random_push_stock

      call timer(TimerIntegrator, TimerOn)
      call timer(TimerNativeStep, TimerOn)
      if (respa .and. npt) then
        npt_bmoment(1:3) = dynvars%barostat_momentum(1:3)
        status = int(gpu_core_npt_respa_vv2(core%ctx,                     &
                     merge(half_dt_long, 0.0_c_double, jstep == multistep),&
                     npt_bmoment))
      else if (respa) then
        status = int(gpu_core_respa_vv2(core%ctx,                         &
                     merge(half_dt_long, 0.0_c_double, jstep == multistep)))
      else if (npt) then
        npt_bmoment(1:3) = dynvars%barostat_momentum(1:3)
        npt_scale(1:3) = exp(-(npt_bmoment(1:3) + sum(npt_bmoment(1:3)) &
                               / ensemble%degree) * half_dt)
        status = int(gpu_core_npt_vv2(core%ctx, npt_scale, npt_bmoment))
      else
        status = int(gpu_core_vv2(core%ctx))
      end if
      if (status /= GcOk) call gpu_core_abort('Gpu_Core_Vverlet> vv2')

      ! the next step's device tick, with nothing reading or moving the
      ! velocities before it (every step ticks, no replica exchange), may
      ! be taken into this VV2's constraint pass
      tick_next = merge(GcConstrainTickNext, 0_c_int32_t,                 &
                        device_thermo .and. i < iend .and.                &
                        dynamics%thermo_period == 1 .and.                 &
                        .not. (respa .or. npt .or. resident))
      call constrain(GcConstrainVV2 + defer_vv2 + tick_next,              &
                     'Gpu_Core_Vverlet> a RATTLE group did not converge')
      if (npt) call gc_check(gpu_core_npt_rattle_end(core%ctx),           &
                             'Gpu_Core_Vverlet> npt rattle')
      ! Stock nve_vv2 adds half of the post-RATTLE constraint virial to
      ! the force virial; VV1's constraint virial remains a separate value.
      ! mtk_barostat_vv2 adds it only without group temperature.
      if (.not. respa .and. .not. npt) then
        do k = 1, 3
          dynvars%virial(k,k) = dynvars%virial(k,k) + 0.5_dp*viri3(k)
        end do
      end if
      ! under gpu_list_guard every step is launched at once: the next
      ! step's list decision reads this step's guard
      graph_kind = GcGraphNone
      if (.not. guard_on)                                                 &
        graph_kind = gpu_core_graph_kind(i+1, istart, iend, dynamics,     &
                                         ensemble, cons_nvt, flat_nve)
      call gc_check(gpu_core_graph_end(core%ctx, graph_kind),             &
                    'Gpu_Core_Vverlet> graph launch')
      call timer(TimerNativeStep, TimerOff)
      call timer(TimerIntegrator, TimerOff)

    end do

    if (respa .or. (resident .and. remd_live)) then
      call native_tail
      if (resident) remd_age = age
      call gpu_core_summary(core, status)
      if (respa) call gpu_core_finalize(core)
      if (present(tail_done)) tail_done = .true.
      ok = .true.
      return
    end if

    if (device_thermo) call thermostat_publish(scale_vel)

    call timer(TimerNativeOutput, TimerOn)
    dynvars%time = dynamics%timestep * real(iend,dp)
    dynvars%step = iend
    call gpu_core_pull(core, domain, status)
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Vverlet> final output boundary pull failed')
    call output_md(output, dynamics, boundary, pairlist, ensemble,        &
                   constraints, dynvars, domain, enefunc, remd)
    ! The restart is already committed.  Refresh all inputs of stock's
    ! extra VV1, then let the existing stock dispatch complete that phase.
    call gpu_core_pull(core, domain, status, with_final=.true.)
    if (status /= GcOk) &
      call gpu_core_abort('Gpu_Core_Vverlet> final VV1 state pull failed')
    call timer(TimerNativeOutput, TimerOff)

    call gpu_core_summary(core, status)
    call gpu_core_finalize(core)

    ok = .true.

  contains

    !------------------------------------------------------------------!
    !  setup_vote: one setup phase's collective verdict; a refusal is  !
    !  printed, the context released and the run handed back to stock !
    !------------------------------------------------------------------!

    subroutine setup_vote(peer, after)

      character(*), intent(in) :: peer
      logical,      intent(in) :: after

      k = status
      call gpu_core_vote(status, nproc_country)
      if (k == GcOk .and. status /= GcOk) core%reason = peer
      if (status == GcOk) return
      if (main_rank .and. after) write(MsgOut,'(A)')                      &
        'Setup_GPU_Core> requested=YES effective=cpu reason=declined '//  &
        'after acceptance: ' // trim(core%reason)
      if (main_rank .and. .not. after) write(MsgOut,'(A)')                &
        'Setup_GPU_Core> requested=YES effective=cpu reason=' //          &
        trim(core%reason)
      call gpu_core_finalize(core)
      ok = .false.
      call timer(TimerNativeSetup, TimerOff)
      call gpu_core_decline(enefunc, core%reason)

    end subroutine setup_vote

    !------------------------------------------------------------------!
    !  output_due: whether a trajectory or restart is written at step  !
    !------------------------------------------------------------------!

    logical function output_due(step)

      integer, intent(in) :: step

      output_due = .false.
      if (dynamics%crdout_period > 0) &
        output_due = output_due .or. mod(step, dynamics%crdout_period) == 0
      if (dynamics%velout_period > 0) &
        output_due = output_due .or. mod(step, dynamics%velout_period) == 0
      if (dynamics%rstout_period > 0) &
        output_due = output_due .or. mod(step, dynamics%rstout_period) == 0

    end function output_due

    !------------------------------------------------------------------!
    !  Small device reads: a kinetic tensor, the half/full pair into   !
    !  s_dynvars, the constraint pass, the group virial and the        !
    !  observable reduction with its energy output                     !
    !------------------------------------------------------------------!

    subroutine kinetic(which, kin3, ekin, what)

      integer(c_int32_t), intent(in)  :: which
      real(c_double),     intent(out) :: kin3(3)
      real(dp),           intent(out) :: ekin
      character(*),       intent(in)  :: what

      call gc_check(gpu_core_kinetic(core%ctx, which, kin3, ekin), what)

    end subroutine kinetic

    subroutine kinetic_pair(which_full, what)

      integer(c_int32_t), intent(in) :: which_full
      character(*),       intent(in) :: what

      call gc_check(gpu_core_kinetic_pair(core%ctx, GcKinGroupVelHalf,    &
                    which_full, kin_half3, ekin_half, kin_full3,          &
                    ekin_full), what)
      call set_kinetic()

    end subroutine kinetic_pair

    subroutine set_kinetic()

      dynvars%kin_half(1:3) = kin_half3(1:3)
      dynvars%kin_full(1:3) = kin_full3(1:3)
      dynvars%ekin_half     = ekin_half
      dynvars%ekin_full     = ekin_full

    end subroutine set_kinetic

    subroutine nvt_observables()

      scale_vel2 = scale_vel * scale_vel
      dynvars%ekin = ((1.0_dp+2.0_dp*scale_vel2)*dynvars%ekin_full &
                    + 2.0_dp*dynvars%ekin_half)/3.0_dp
      dynvars%kin(1:3) = 0.5_dp*(1.0_dp+scale_vel2) &
                         *dynvars%kin_full(1:3) + dynvars%kin_half(1:3)

    end subroutine nvt_observables

    subroutine constrain(mode, what)

      integer(c_int32_t), intent(in) :: mode
      character(*),       intent(in) :: what

      integer(c_int32_t) :: st

      call timer(TimerConstraint, TimerOn)
      st = gpu_core_constrain(core%ctx, mode, real(dt, c_double), viri3, nfail)
      call timer(TimerConstraint, TimerOff)
      call gc_check(st, what)

    end subroutine constrain

    subroutine add_group_virial(what)

      character(*), intent(in) :: what

      ! compute_virial_group writes the negative of sum (r - r_cm)*f over
      ! the rigid groups into its virial argument, which nve_vv1 adds to
      ! the force virial: the group tensor is that negative sum
      call gc_check(gpu_core_group_virial(core%ctx, viri3), what)
      do k = 1, 3
        dynvars%virial_group(k,k) = -viri3(k)
        dynvars%virial(k,k) = dynvars%virial(k,k) + dynvars%virial_group(k,k)
      end do

    end subroutine add_group_virial

    subroutine energy_output(what)

      character(*), intent(in) :: what

      call gc_check(gpu_core_dynvars_sums(core%ctx, c_loc(rmsg_local),    &
                    c_loc(ekin_sum_local)), what)
      call compute_dynvars(enefunc, dynamics, boundary, ensemble, domain, &
                           dynvars, real(rmsg_local,dp),                 &
                           real(ekin_sum_local,dp))
      call output_dynvars(output, enefunc, dynvars, ensemble)

    end subroutine energy_output

    !------------------------------------------------------------------!
    !  host_scale: stock's velocity scale factor from ekin, on rank 0's!
    !  draw for every rank. draw: 0 none (drawn by the caller), 1 a    !
    !  gauss number for Bussi only, 2 one in every case                !
    !------------------------------------------------------------------!

    subroutine host_scale(ndeg, dtt, ekin_in, scale_out, draw)

      integer(iintegers), intent(in)  :: ndeg
      real(wip),          intent(in)  :: dtt
      real(dp),           intent(in)  :: ekin_in
      real(wip),          intent(out) :: scale_out
      integer,            intent(in)  :: draw

      scale_out = 1.0_wip
      if (draw == 2 .or. (draw == 1 .and.                                 &
                          ensemble%tpcontrol == TpcontrolBussi))          &
        rr = real(random_get_gauss(), wip)
      if (ensemble%tpcontrol == TpcontrolBussi) then
        call gpu_core_scale_bussi(ndeg, dtt, ensemble%tau_t/AKMA_PS,      &
                                  ensemble%temperature, ekin_in, rr,     &
                                  scale_out)
      else if (ensemble%tpcontrol == TpcontrolBerendsen) then
        call gpu_core_scale_berendsen(ndeg, dtt, ensemble%tau_t/AKMA_PS,  &
                                      ensemble%temperature, ekin_in,     &
                                      scale_out)
      else if (ensemble%tpcontrol == TpcontrolNHC) then
        call gpu_core_scale_nhc(ndeg, dtt, ensemble%tau_t/AKMA_PS,        &
                                ensemble%temperature, ekin_in, ensemble, &
                                dynvars, scale_out)
      end if
#ifdef HAVE_MPI_GENESIS
      call mpi_bcast(scale_out, 1, mpi_wip_real, 0, mpi_comm_country, &
                     ierror)
#endif

    end subroutine host_scale

    !------------------------------------------------------------------!
    !  cons_nvt_vv1: stock's constrained flat-NVT VV1                  !
    !  (vel_rescaling_thermostat_vv1_cons) on the device state, with   !
    !  rigid bonds and no group temperature. One gauss draw every step;!
    !  on a thermostat tick four passes of (scaled kick from the       !
    !  reference + drift + VV1 SHAKE), the factor re-derived from each !
    !  pass's post-SHAKE kinetic and the constraint virial taken from  !
    !  the last pass; ekin/kin published only on a tick; the reference !
    !  kinetic refreshed on the step after a tick and at step 1.       !
    !------------------------------------------------------------------!

    subroutine cons_nvt_vv1(istep)

      integer, intent(in) :: istep

      integer        :: iter, maxiter
      real(c_double) :: kin_ref3(3), kin3(3)
      real(dp)       :: ekin_ref, ekin3
      logical        :: tick

      tick    = mod(istep-1, dynamics%thermo_period) == 0 .and. istep > 1
      maxiter = merge(4, 1, tick)
      scale_vel   = 1.0_wip
      kin_ref3(1:3) = 0.0_dp
      ekin_ref      = 0.0_dp

      rr = real(random_get_gauss(), wip)

      if (tick) call kinetic(GcKinFlatVel, kin_ref3, ekin_ref,            &
                    'Gpu_Core_Vverlet> constrained NVT reference kinetic')

      if (ensemble%tpcontrol == TpcontrolNHC) &
        dynvars%nh_velocity_ref(1:5) = dynvars%nh_velocity(1:5)

      do iter = 1, maxiter

        dynvars%kin_full(1:3) = kin_ref3(1:3) * scale_vel*scale_vel
        dynvars%ekin_full     = ekin_ref * scale_vel*scale_vel

        call gc_check(gpu_core_vv1(core%ctx, real(scale_vel, c_double),   &
                      GcVv1FromRef), 'Gpu_Core_Vverlet> vv1')
        call constrain(GcConstrainVV1, 'Gpu_Core_Vverlet> a constraint '//&
                       'group did not converge; see GPU_Core_Constraint')
        if (iter == maxiter) then
          do k = 1, 3
            dynvars%virial_const(k,k) = viri3(k)
            dynvars%virial(k,k) = dynvars%virial(k,k) + viri3(k)
          end do
        end if

        if (tick) then

          call kinetic(GcKinFlatVel, kin3, ekin3,                         &
                       'Gpu_Core_Vverlet> constrained NVT half kinetic')
          dynvars%kin_half(1:3) = kin3(1:3)
          dynvars%ekin_half     = ekin3
          dynvars%ekin = (dynvars%ekin_full + dynvars%ekin_half           &
                          + dynvars%ekin_ref) / 3.0_dp
          dynvars%kin(1:3) = 0.5_dp * (dynvars%kin_ref(1:3)               &
                                       + dynvars%kin_half(1:3))
          if (ensemble%tpcontrol == TpcontrolNHC) &
            dynvars%nh_velocity(1:5) = dynvars%nh_velocity_ref(1:5)
          call host_scale(num_degree, real(dt_therm,wip), dynvars%ekin,   &
                          scale_vel, 0)
          if (iter == maxiter) then
            dynvars%kin_ref(1:3) = dynvars%kin_half(1:3)
            dynvars%ekin_ref     = dynvars%ekin_half
          end if

        else if (mod(istep-2, dynamics%thermo_period) == 0 .or.           &
                 istep == 1) then

          call kinetic(GcKinFlatVel, kin3, ekin3,                         &
                       'Gpu_Core_Vverlet> constrained NVT reference kinetic')
          dynvars%kin_ref(1:3) = kin3(1:3)
          dynvars%ekin_ref     = ekin3

        end if

      end do

    end subroutine cons_nvt_vv1

    !----------------------------------------------------------------------!
    !  thermostat_start: the device thermostat's constants, in             !
    !  vel_scale_bussi / _berendsen / _nhc's own expressions, and the chain!
    !  the host holds.                                                     !
    !----------------------------------------------------------------------!

    subroutine thermostat_start

      real(wip)      :: tau_t, kbt, w(3), dt_small, dt_1
      real(c_double) :: nh_dt(12), nh_mass(GcNhcMax), nh_vel(GcNhcMax)
      real(c_double) :: nh_force(GcNhcMax), nh_coef(GcNhcMax)
      real(c_double) :: kin6(6)
      integer        :: j, n, tkind

      tkind = GcThermoNone
      if (ensemble%tpcontrol == TpcontrolBerendsen) tkind = GcThermoBerend
      if (ensemble%tpcontrol == TpcontrolBussi)     tkind = GcThermoBussi
      if (ensemble%tpcontrol == TpcontrolNHC)       tkind = GcThermoNHC
      tau_t = ensemble%tau_t/AKMA_PS
      kbt   = KBOLTZ * ensemble%temperature
      n     = min(ensemble%nhchain, GcNhcMax)

      w(1) = 1.0_dp / (2.0_dp - 2.0_dp**(1.0_dp/3.0_dp))
      w(3) = w(1)
      w(2) = 1.0_dp - w(1) - w(3)
      dt_small = real(dt_therm,wip) / real(max(ensemble%nhmultistep,1), wip)
      do j = 1, 3
        dt_1 = w(j) * dt_small
        nh_dt(4*j-3) = dt_1
        nh_dt(4*j-2) = dt_1 * 0.5_dp
        nh_dt(4*j-1) = nh_dt(4*j-2) * 0.5_dp
        nh_dt(4*j)   = nh_dt(4*j-1) * 0.5_dp
      end do
      if (tkind == GcThermoNHC) then
        dynvars%nh_mass(2:n) = kbt * tau_t*tau_t
        dynvars%nh_mass(1)   = real(num_degree,wip) * dynvars%nh_mass(2)
      end if
      nh_mass(1:n)  = dynvars%nh_mass(1:n)
      nh_vel(1:n)   = dynvars%nh_velocity(1:n)
      nh_force(1:n) = dynvars%nh_force(1:n)
      nh_coef(1:n)  = dynvars%nh_coef(1:n)
      kin6(1:3) = dynvars%kin_half(1:3)
      kin6(4:6) = dynvars%kin_full(1:3)

      status = int(gpu_core_thermostat_init(core%ctx, int(tkind,c_int32_t),&
                   int(ensemble%nhchain, c_int32_t),                     &
                   int(ensemble%nhmultistep, c_int32_t),                 &
                   real(num_degree, c_double), real(KBOLTZ, c_double),   &
                   real(ensemble%temperature, c_double),                 &
                   real(real(dt_therm,wip)/tau_t, c_double),             &
                   real(exp(-real(dt_therm,wip)/tau_t), c_double),       &
                   real(kbt, c_double), nh_dt, nh_mass, nh_vel,          &
                   nh_force, nh_coef, kin6))
      if (status /= GcOk) &
        call gpu_core_abort('Gpu_Core_Vverlet> thermostat setup: ' //    &
                            trim(gc_status_text(status)))

    end subroutine thermostat_start

    !------------------------------------------------------------------------!
    !  bussi_draws: the (rr, sum_gauss) pairs of the Bussi ticks from istep  !
    !  on, drawn in stock's order and uploaded in one call. The batch stops  !
    !  before the next restart boundary and at the last step, so the         !
    !  generator state the restart writer stocks and the one the caller      !
    !  continues with are stock's. Every rank draws; rank 0's draws are used.!
    !------------------------------------------------------------------------!

    subroutine bussi_draws(istep)

      integer, intent(in) :: istep
      integer             :: n, j, last

      last = iend
      if (dynamics%rstout_period > 0)                                     &
        last = min(last, ((istep-1)/dynamics%rstout_period + 1)          &
                         * dynamics%rstout_period)
      n = 0
      j = istep
      do while (n < GcThermoDraws .and. j <= last)
        n = n + 1
        rr = real(random_get_gauss(), wip)
        thermo_draw(1,n) = rr
        thermo_draw(2,n) = sum_gauss(num_degree-1)
        j = j + dynamics%thermo_period
      end do
#ifdef HAVE_MPI_GENESIS
      call mpi_bcast(thermo_draw, 2*n, mpi_real8, 0, mpi_comm_country,  &
                     ierror)
#endif
      call gc_check(gpu_core_thermostat_draws(core%ctx, int(n,c_int32_t), &
                    thermo_draw), 'Gpu_Core_Vverlet> thermostat draws')
      draws_left = n

    end subroutine bussi_draws

    !------------------------------------------------------------------!
    !  thermostat_publish: the last tick's kinetic tensors, its scale  !
    !  and the chain, into s_dynvars (synchronises).                   !
    !------------------------------------------------------------------!

    subroutine thermostat_publish(scale)

      real(wip), intent(out) :: scale

      real(c_double) :: sc, nh_vel(GcNhcMax), nh_force(GcNhcMax)
      real(c_double) :: nh_coef(GcNhcMax)
      integer        :: n

      call gc_check(gpu_core_thermostat_state(core%ctx, kin_half3,        &
                    ekin_half, kin_full3, ekin_full, sc, nh_vel,          &
                    nh_force, nh_coef), 'Gpu_Core_Vverlet> thermostat state')
      call set_kinetic()
      scale = real(sc, wip)
      if (ensemble%tpcontrol == TpcontrolNHC) then
        n = min(ensemble%nhchain, GcNhcMax)
        dynvars%nh_velocity(1:n) = nh_vel(1:n)
        dynvars%nh_force(1:n)    = nh_force(1:n)
        dynvars%nh_coef(1:n)     = nh_coef(1:n)
      end if

    end subroutine thermostat_publish

    !------------------------------------------------------------------------!
    !  npt_vv1: mtk_barostat_vv1 with the group temperature convention, of   !
    !  sp_md_vverlet.fpp or, under r-RESPA, of sp_md_respa.fpp. The scalars  !
    !  follow stock line by line; the particle arrays stay on the device.    !
    !  Velocity-Verlet folds the two thermostat halves (each a group centre- !
    !  of-mass scale) into the sweep's vel_scale; r-RESPA applies each where !
    !  stock applies it.                                                     !
    !------------------------------------------------------------------------!

    subroutine npt_vv1(istep)

      integer, intent(in) :: istep

      logical        :: calc_elec_long, calc_thermostat, calc_barostat
      integer        :: first
      real(wip)      :: d_ndegf, volume, press0, tau_p, dt_baro
      real(wip)      :: press(3), pressxyz, pressxy, s_therm, gr
      real(wip)      :: scale_b(3)
      real(dp)       :: virial_sum(3)
      real(c_double) :: size_scale(3), vel_scale(3), viri_grp(3)
      real(wip), pointer :: bmoment(:)

      bmoment => dynvars%barostat_momentum
      d_ndegf = real(num_degree, wip)
      press0  = ensemble%pressure * ATMOS_P
      tau_p   = ensemble%tau_p / AKMA_PS
      dt_baro = real(dt, wip) * real(dynamics%baro_period, wip)

      boundary%box_size_x_ref = boundary%box_size_x
      boundary%box_size_y_ref = boundary%box_size_y
      boundary%box_size_z_ref = boundary%box_size_z
      volume = boundary%box_size_x * boundary%box_size_y * boundary%box_size_z

      first = merge(istart, 1, respa)
      calc_elec_long  = respa .and. mod(istep-1, multistep) == 0
      calc_thermostat = mod(istep-1, dynamics%thermo_period) == 0 .and. &
                        istep > first
      calc_barostat   = mod(istep-1, dynamics%baro_period) == 0 .and.   &
                        istep > first

      if (istep == 1) then
        ensemble%degree = d_ndegf + 3.0_wip
        if (ensemble%isotropy == IsotropyXY_Fixed) &
          ensemble%degree = d_ndegf + 1.0_wip
        ensemble%kinetic = 0.5_wip*KBOLTZ*ensemble%temperature*ensemble%degree
        ensemble%pmass   = ensemble%degree*KBOLTZ*ensemble%temperature &
                         * tau_p*tau_p
      end if

      if (respa) then
        ! the second half of the previous outer step's barostat scale
        if (calc_elec_long .and. istep > 1) then
          gr = bmoment(1) + bmoment(2) + bmoment(3)
          gr = gr / ensemble%degree
          scale_b(1:3) = bmoment(1:3) + gr
          vel_scale(1:3) = exp(-scale_b(1:3)*real(half_dt_long, wip))
          call gc_check(gpu_core_npt_group_scale(core%ctx, vel_scale),    &
                        'Gpu_Core_Npt> group scale')
        end if
      else
        ! stock draws one number here on every step, used or not
        rr = real(random_get_gauss(), wip)
      end if

      s_therm = 1.0_wip
      if (calc_thermostat) then
        if (respa) then
          call gc_check(gpu_core_respa_half(core%ctx),                    &
                        'Gpu_Core_Npt> half velocity')
        else
          call gc_check(gpu_core_nvt_half(core%ctx),                      &
                        'Gpu_Core_Npt> half velocity')
        end if
        call kinetic_pair(GcKinGroupVel, 'Gpu_Core_Npt> kinetic pair')
        dynvars%ekin = ekin_full + 2.0_dp*ekin_half/3.0_dp
        dynvars%ekin = dynvars%ekin &
                     + 0.5_dp*ensemble%pmass*dot_product(bmoment,bmoment)
        call npt_thermostat_half(.true., s_therm)
      end if

      ! r-RESPA: the force virial of an outer step already holds the
      ! reciprocal part, which stock adds here from virial_long
      if (calc_barostat) then
        call gc_check(gpu_core_group_virial(core%ctx, viri_grp),          &
                      'Gpu_Core_Npt> group virial')
        ! compute_virial_group's tensor is the negative sum
        dynvars%virial_group(1:3,1:3) = 0.0_dp
        do k = 1, 3
          dynvars%virial_group(k,k) = -viri_grp(k)
          virial_sum(k) = dynvars%virial(k,k) + dynvars%virial_group(k,k)
          dynvars%virial(k,k) = virial_sum(k)
        end do
        ! the bonded virial rides this allreduce exactly (gpu_core_collect)
        call gpu_core_npt_virial_sum(virial_sum)
        press(1:3) = (dynvars%kin(1:3) + virial_sum(1:3))/volume
        pressxyz = (press(1)+press(2)+press(3))/3.0_dp
        pressxy  = (press(1)+press(2))/2.0_dp
        if (respa) then
          call gpu_core_update_barostat_mtk(ensemble, press, pressxyz,     &
                                            pressxy, press0, volume,       &
                                            d_ndegf, ensemble%pmass,       &
                                            dt_baro, dynvars%ekin, bmoment)
        else
          call gpu_core_update_barostat(ensemble, boundary, press,         &
                                        pressxyz, pressxy, press0, volume, &
                                        d_ndegf, ensemble%pmass, dt_baro,  &
                                        dynvars%ekin, bmoment)
        end if
      end if

      if (calc_thermostat) then
        dynvars%ekin = dynvars%ekin &
                     + 0.5_dp*ensemble%pmass*dot_product(bmoment,bmoment)
        call npt_thermostat_half(.false., s_therm)
        dynvars%ekin = dynvars%ekin &
                     + 0.5_dp*ensemble%pmass*dot_product(bmoment,bmoment)
      end if

      if (respa) then
        ! on an outer step the first half of this step's barostat scale and
        ! the long kick, then the short kick and compute_vv1_coord_group
        vel_scale(1:3) = 1.0_c_double
        if (calc_elec_long) then
          gr = bmoment(1) + bmoment(2) + bmoment(3)
          scale_b(1:3) = bmoment(1:3) + gr/ensemble%degree
          vel_scale(1:3) = exp(-scale_b(1:3)*real(half_dt_long, wip))
        end if
        size_scale(1:3) = exp(bmoment(1:3)*real(half_dt, wip))
        call gc_check(gpu_core_npt_respa_vv1(core%ctx, vel_scale,          &
                      merge(half_dt_long, 0.0_c_double, calc_elec_long),   &
                      size_scale), 'Gpu_Core_Npt> vv1')
      else
        size_scale(1:3) = exp(bmoment(1:3)*dt)
        gr = bmoment(1) + bmoment(2) + bmoment(3)
        scale_b(1:3) = bmoment(1:3) + gr/ensemble%degree
        vel_scale(1:3) = exp(-scale_b(1:3)*half_dt) * s_therm
        call gc_check(gpu_core_npt_vv1(core%ctx, size_scale, vel_scale),   &
                      'Gpu_Core_Npt> vv1')
      end if

      ! stock r-RESPA keeps the constraint virial out of the virial
      call constrain(GcConstrainVV1 + merge(0_c_int32_t, GcConstrainDefer, &
                     want_energy), 'Gpu_Core_Npt> a constraint group did '//&
                     'not converge; see GPU_Core_Constraint')
      do k = 1, 3
        dynvars%virial_const(k,k) = viri3(k)
      end do

      call npt_box(istep)


    end subroutine npt_vv1

    !------------------------------------------------------------------!
    !  npt_thermostat_half: one thermostat half of mtk_barostat_vv1:   !
    !  the scale for half a thermostat period with the degree that     !
    !  counts the barostat (velocity-Verlet draws a gauss number       !
    !  unconditionally, r-RESPA for Bussi only), applied to the        !
    !  barostat momentum and the full kinetic; velocity-Verlet folds it!
    !  into s_therm, r-RESPA scales the groups on the device           !
    !------------------------------------------------------------------!

    subroutine npt_thermostat_half(first_half, s_therm)

      logical,   intent(in)    :: first_half
      real(wip), intent(inout) :: s_therm

      real(wip)      :: s
      real(c_double) :: vel_scale(3)

      call host_scale(int(ensemble%degree, iintegers),                    &
                      real(dt_therm, wip) / 2.0_wip, dynvars%ekin, s,     &
                      merge(1, 2, respa))
      if (respa) then
        vel_scale(1:3) = real(s, c_double)
        call gc_check(gpu_core_npt_group_scale(core%ctx, vel_scale),      &
                      'Gpu_Core_Npt> group scale')
      else
        s_therm = s_therm * s
      end if
      dynvars%barostat_momentum(1:3) = dynvars%barostat_momentum(1:3) * s
      dynvars%kin_full(1:3) = dynvars%kin_full(1:3)*s*s
      if (first_half) &
        dynvars%kin(1:3) = dynvars%kin_full(1:3) + dynvars%kin_half(1:3)
      dynvars%ekin = 0.5_dp*(dynvars%kin_full(1) + dynvars%kin_full(2)    &
                             + dynvars%kin_full(3))                       &
                   + 2.0_dp*dynvars%ekin_half/3.0_dp

    end subroutine npt_thermostat_half

    !------------------------------------------------------------------!
    !  npt_box: the box of t+dt from the barostat momentum, and every  !
    !  image offset with it; on a barostat step the reciprocal sum     !
    !  takes the new box (compute_energy's npt1)                       !
    !------------------------------------------------------------------!

    subroutine npt_box(istep)

      integer, intent(in) :: istep

      real(wip)      :: scale_b(3), box(3)

      scale_b(1:3) = exp(dynvars%barostat_momentum(1:3)*dt)
      box(1) = scale_b(1) * boundary%box_size_x_ref
      box(2) = scale_b(2) * boundary%box_size_y_ref
      box(3) = scale_b(3) * boundary%box_size_z_ref
#ifdef HAVE_MPI_GENESIS
      call mpi_bcast(box, 3, mpi_wip_real, 0, mpi_comm_country, ierror)
#endif
      boundary%box_size_x  = box(1)
      boundary%box_size_y  = box(2)
      boundary%box_size_z  = box(3)
      boundary%cell_size_x = box(1) / real(boundary%num_cells_x,wip)
      boundary%cell_size_y = box(2) / real(boundary%num_cells_y,wip)
      boundary%cell_size_z = box(3) / real(boundary%num_cells_z,wip)
      domain%system_size(1:3) = box(1:3)
      status = int(gpu_core_box_scale(core%ctx,                            &
                   real(scale_b, c_double), real(box, c_double),           &
                   merge(1_c_int32_t, 0_c_int32_t,                         &
                         mod(istep, dynamics%baro_period) == 0)))
      if (status /= GcOk) &
        call gpu_core_abort('Gpu_Core_Npt> box scale: ' // &
                            trim(gc_status_text(status)))

    end subroutine npt_box

    !------------------------------------------------------------------------!
    !  native_tail: the stock loop's tail for a context that stays on the    !
    !  device (r-RESPA, and the resident temperature-REMD context): output_md!
    !  at iend; one extra VV1 of step iend+1 (with its thermostat draw); then!
    !  coord_vel_ref, which keeps that trial velocity as the half velocity   !
    !  and restores velocity, coordinates and the NHC chain                  !
    !  (gpu_core_remd_rollback); then compute_dynvars/output_dynvars. Under  !
    !  REMD the particle state crosses to the host only when output_md is    !
    !  due to write it; r-RESPA writes its final output at iend always.      !
    !------------------------------------------------------------------------!

    subroutine native_tail

      call timer(TimerNativeOutput, TimerOn)
      dynvars%time = dynamics%timestep * real(iend,dp)
      dynvars%step = iend
      ! output_md writes nothing unless one of its periods is due
      ! (sp_output.fpp (output_md)), so without a pull it reads nothing
      if (respa .or. output_due(iend)) then
        call gpu_core_pull(core, domain, status)
        if (status /= GcOk) &
          call gpu_core_abort('Gpu_Core_Vverlet> final output boundary '// &
                              'pull failed')
      end if
      call output_md(output, dynamics, boundary, pairlist, ensemble,      &
                     constraints, dynvars, domain, enefunc, remd)
      call timer(TimerNativeOutput, TimerOff)

      i = iend + 1
      want_energy = (mod(i-1, dynamics%eneout_period) == 0)
      call gc_check(gpu_core_step_begin(core%ctx, int(i, c_int64_t),     &
                    merge(1_c_int32_t, 0_c_int32_t, want_energy)),       &
                    'Gpu_Core_Vverlet> step_begin')

      if (respa .and. npt) then
        ! mtk_barostat_vv1: its box change and barostat momentum outlive
        ! coord_vel_ref's rollback, as in stock
        call npt_vv1(i)
      else if (respa) then
        scale_vel = 1.0_wip
        if (ensemble%ensemble == EnsembleNVE) then
          if (want_energy) then
            call kinetic_pair(GcKinGroupVel, 'Gpu_Core_Respa> kinetic')
            dynvars%ekin     = ekin_full + 2.0_dp*ekin_half/3.0_dp
            dynvars%kin(1:3) = kin_full3(1:3) + kin_half3(1:3)
          end if
        else
          if (mod(i-1, dynamics%thermo_period) == 0 .and. i > istart) then
            call gc_check(gpu_core_respa_half(core%ctx),                  &
                          'Gpu_Core_Respa> half velocity')
            call kinetic_pair(GcKinGroupVelRef, 'Gpu_Core_Respa> kinetic')
            call host_scale(num_degree, real(dt_therm,wip),               &
                            ekin_full + 2.0_dp*ekin_half/3.0_dp,          &
                            scale_vel, 1)
          end if
          call nvt_observables()
        end if
        call gc_check(gpu_core_respa_vv1(core%ctx,                        &
                      real(scale_vel, c_double), half_dt_long),           &
                      'Gpu_Core_Respa> vv1')
        call constrain(GcConstrainVV1, 'Gpu_Core_Respa> a constraint '//  &
                       'group did not converge; see GPU_Core_Constraint')
        do k = 1, 3
          dynvars%virial_const(k,k) = viri3(k)
        end do
        if (want_energy) call add_group_virial('Gpu_Core_Respa> group virial')
      else
        scale_vel = 1.0_wip
        if (mod(i-1, dynamics%thermo_period) == 0 .and. i > 1) then
          call gc_check(gpu_core_nvt_half(core%ctx),                      &
                        'Gpu_Core_Remd> NVT half velocity')
          call kinetic(merge(GcKinGroupVelHalf, GcKinFlatVelHalf,         &
                             ensemble%group_tp), kin_half3, ekin_half,    &
                       'Gpu_Core_Remd> NVT half kinetic')
          call kinetic(merge(GcKinGroupVelRef, GcKinFlatVelRef,           &
                             ensemble%group_tp), kin_full3, ekin_full,    &
                       'Gpu_Core_Remd> NVT full kinetic')
          call set_kinetic()
          call host_scale(num_degree, real(dt_therm,wip),                 &
                          ekin_full + 2.0_dp*ekin_half/3.0_dp, scale_vel, 1)
        end if
        call nvt_observables()
        call gc_check(gpu_core_vv1(core%ctx, real(scale_vel, c_double),   &
                      GcVv1Kick), 'Gpu_Core_Remd> vv1')
        call constrain(GcConstrainVV1, 'Gpu_Core_Remd> a constraint '//   &
                       'group did not converge; see GPU_Core_Constraint')
        do k = 1, 3
          dynvars%virial_const(k,k) = viri3(k)
          dynvars%virial(k,k) = dynvars%virial(k,k) + viri3(k)
        end do
        if (want_energy .and. ensemble%group_tp) &
          call add_group_virial('Gpu_Core_Remd> group virial')
      end if

      call gc_check(gpu_core_remd_rollback(core%ctx),                     &
                    'Gpu_Core_Vverlet> rollback')
      if (.not. respa .or. ensemble%tpcontrol == TpcontrolNHC) &
        dynvars%nh_velocity(1:5) = dynvars%nh_velocity_ref(1:5)

      call timer(TimerNativeEnergy, TimerOn)
      call energy_output('Gpu_Core_Vverlet> scalar observable reduction')
      call timer(TimerNativeEnergy, TimerOff)

    end subroutine native_tail

  end subroutine gpu_core_vverlet

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gc_check
  !> @brief        stop on a refused native call
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gc_check(status, message)

    integer(c_int32_t),      intent(in) :: status
    character(*),            intent(in) :: message

    if (status /= GcOk) call gpu_core_abort(message)

  end subroutine gc_check

  !======1=========2=========3=========4=========5=========6=========7=========8
  !
  !  Subroutine    gpu_core_abort
  !> @brief        a native failure stops every rank: a refusal on some ranks
  !!               must never leave the others waiting in a collective, and
  !!               error_msg's exit(1) ends only the calling process
  !
  !======1=========2=========3=========4=========5=========6=========7=========8

  subroutine gpu_core_abort(message)

    character(*),            intent(in)    :: message

#ifdef HAVE_MPI_GENESIS
    integer                  :: ierror

    write(ErrOut,'(A,A12,I5)') trim(message), '  rank_no = ', my_world_rank
    flush(ErrOut)
    call mpi_abort(mpi_comm_world, 1, ierror)
#endif
    call error_msg(message)

  end subroutine gpu_core_abort

end module sp_gpu_core_step_mod
