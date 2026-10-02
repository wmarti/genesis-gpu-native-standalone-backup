.. highlight:: bash
.. _gpu_native:

=======================================================================
Native GPU core (SPDYN)
=======================================================================

Overview
=======================================================================

In the standard GPU build of **SPDYN**, only the non-bonded real-space
interaction is computed on the GPU; coordinates and forces are copied
between the host and the device at every step.  The native GPU core
instead keeps the whole MD state on the GPU for the length of the run:
coordinates, velocities, forces, the pair list, the bonded terms, the PME
reciprocal-space sum, the constraints, the integrator and the thermostat.
The host keeps GENESIS's own setup, input and output, and the barostat
scalars.  Between output steps no particle data is copied to the host.
Each rank drives one GPU; neighbouring ranks exchange halo and migration
data directly between GPUs (peer-to-peer or CUDA-aware MPI on a node,
MPI between nodes).

The native core is requested per run with ``gpu_resident`` in the
[DYNAMICS] section.  If the requested simulation is outside its supported
span, all ranks together fall back to the standard path of the same
binary, and the log says so (see below).

Supported simulations
=======================================================================

* force field **CHARMM**, ``electrostatic = PME`` (table-based real space,
  GPU reciprocal space) or ``CUTOFF``; **AMBER** and **GROAMBER** with
  ``electrostatic = PME`` (GROAMBER without Ryckaert-Bellemans dihedrals
  or harmonic impropers);
* ``integrator = VVER``, and ``VRES`` (r-RESPA) for NVE and NVT with
  ``rigid_bond = YES`` and ``group_tp = YES`` when ``eneout_period`` and
  ``nsteps`` are multiples of ``elec_long_period``;
* ``ensemble = NVE``, ``NVT`` (``tpcontrol = BERENDSEN``, ``BUSSI`` or
  ``NHC``), and ``NPT``, ``NPAT`` or ``NPgT`` with the MTK barostat
  (``BERENDSEN``, ``BUSSI`` or ``NHC``; ``integrator = VVER``,
  ``group_tp = YES``, ``rigid_bond = YES``);
* SHAKE/RATTLE for hydrogen groups and SETTLE for rigid water;
* hydrogen mass repartitioning (``hydrogen_mr = YES`` in [DYNAMICS], or a
  topology whose masses are already repartitioned);
* positional restraints and [LOCAL_RESTRAINT] bonds and angles;
* temperature REMD (the replicas each run the native core);
* any number of MPI processes, one GPU per process, on one or more nodes.

Other settings (other force fields, LJ-PME and dispersion correction,
FEP, GaMD, RPATH, other restraint types, targeted and steered MD, TIP4P
virtual sites, the Langevin barostat, NPT in
REMD, triclinic boxes) are not supported by the native core and run on the
standard path.  The native core requires a double-precision build
(``--enable-double``, the default).

Fallback
=======================================================================

The decision is collective: either all ranks run the native core or all
fall back.  Every run reports its path on rank 0 in one line:

::

  Setup_GPU_Core> requested=YES effective=native precision=FP64 reason=eligible
  Setup_GPU_Core> requested=YES effective=cpu reason=<why the core declined>

A declined run logs its reason in that line and falls back to the standard
path.  Settings that only the native core runs (``electrostatic = CUTOFF``,
``nonbond_precision = MIXED``) stop with that reason instead.
At the end of a native run ``GPU_Core_Summary>`` lines report the number of
native steps, the pair-list rebuilds and the largest atomic displacement
between rebuilds.

Keywords
=======================================================================

[DYNAMICS] section
-----------------------------------------------------------------------

**gpu_resident** *YES / NO* (**SPDYN** GPU build only)

  **Default : NO**

  Run the dynamics on the native GPU core.  *NO* selects the standard
  path unchanged.

**gpu_step_graph** *YES / NO* (**SPDYN** GPU build only)

  **Default : NO**

  Replay each plain step of a multi-rank native run (no energy, output,
  pair-list rebuild or barostat work) as one CUDA graph.  The results are
  bitwise identical to *NO*.

**gpu_list_guard** *YES / NO* (**SPDYN** GPU build only)

  **Default : NO**

  Let a pair list live up to ``nbupdate_period`` steps, cutting it short
  when one atom may have moved the whole list buffer (``pairlistdist``
  minus the cutoff).  Choose ``pairlistdist`` and ``nbupdate_period``
  together so that the missed-pair error stays no larger than that of the
  standard 13.5 A list rebuilt every 10 steps.  *NO* rebuilds every
  ``nbupdate_period`` steps, as the standard path does.  Not used with
  r-RESPA or NPT.

**gpu_route_mesh**, **gpu_route_coord**, **gpu_route_force** *MPI / THREAD* (**SPDYN** GPU build only)

  **Default : MPI**

  Inter-node transport of the PME mesh transposes, the coordinate halo
  and the force halo.  *MPI* uses CUDA-aware MPI, or host staging where
  that is not available; *THREAD* hands the inter-node messages to a
  transport thread (MPI_THREAD_MULTIPLE).  The route does not change the
  results.  Exchanges within a node always use peer-to-peer copies or
  CUDA-aware MPI.

[ENERGY] section
-----------------------------------------------------------------------

**nonbond_precision** *DOUBLE / MIXED* (**SPDYN** native GPU core only)

  **Default : DOUBLE**

  Arithmetic of the native core's PME non-bonded terms.

  *DOUBLE*: every quantity is computed in double precision.  Forces,
  energies and virials are accumulated in 64-bit fixed point, so that for
  a given number of processes the trajectory is bitwise reproducible from
  run to run and independent of GPU thread scheduling.  Energies agree
  with the standard path to the regression tolerance (relative 1.0e-6).

  *MIXED*: the real-space pair term, the bonded, 1-4 and excluded-pair
  terms, the PME mesh and the water constraints are evaluated in single
  precision; the pair list, the accumulation (64-bit fixed point) and the
  integration stay in double precision, so a MIXED run is also bitwise
  reproducible.  The regression tests compare MIXED runs at the
  single-precision tolerance (3.0e-5).  The log reports
  ``precision = mixed``.  *MIXED* requires ``gpu_resident = YES`` and
  ``electrostatic = PME``.

**ewald_evaluation** *TABLE / ANALYTIC* (**SPDYN** native GPU core, MIXED only)

  **Default : ANALYTIC** where the table has the closed form (PME with
  the CHARMM switch, ``vdw_force_switch = NO``) and the polynomial
  reaches single-precision accuracy over the cutoff, otherwise **TABLE**

  How the MIXED pair term evaluates Ewald real space and Lennard-Jones.
  *TABLE* reads GENESIS's lookup table (``table_density``) as single
  precision records.  *ANALYTIC* evaluates the closed form with a
  polynomial erfc; it has no interpolation error, and the core checks
  that it reproduces the table's function.  An explicit *ANALYTIC* on
  another form stops the run, and so does *ANALYTIC* with
  ``nonbond_precision = DOUBLE``.  The 1-4 terms read the table in every
  mode.

**pme_grid** *INPUT / ACCURACY* (**SPDYN** only)

  **Default : INPUT**

  *INPUT* uses ``pme_ngrid_x/y/z`` (or ``pme_max_spacing``) and
  ``pme_alpha`` as given.  *ACCURACY* keeps the estimated RMS force error
  of the input mesh and ``pme_alpha`` and chooses the smallest mesh (sizes
  with factors 2, 3, 5 and 7) and the Ewald parameter that reach it.  The
  choice is printed in a ``Select_Pme_Grid>`` line.  ``pme_nspline`` must
  be 4 or 6.

[BOUNDARY] section
-----------------------------------------------------------------------

**domain_select** *STOCK / HALO* (**SPDYN** only)

  **Default : HALO** (GPU build), **STOCK** (otherwise)

  How the domain grid is chosen when ``domain_x``, ``domain_y`` and
  ``domain_z`` are all 0; given values are always used as they are.
  *STOCK* is GENESIS's own rule.  *HALO* considers the grids with the
  process count, domains at least the minimum width on every axis, and
  x divided before y and y before z, and takes the one whose domains
  import the smallest halo volume.  The grid only changes
  runs with more than one process.

Building
=======================================================================

The native core is part of every GPU build:

::

  $ ./configure --enable-gpu --enable-double --with-gpuarch=sm_80 CC=mpicc FC=mpif90

``--with-gpuarch`` takes the target GPU architecture; nvcc
``--generate-code`` options may follow it to build one binary for several
architectures.  Further options:

``--enable-gpu-profile``
  build with per-kernel event timing for profiling.

The regression tests of the native core are in
``tests/regression_test/test_gpu_native.py``:

::

  $ ./test_gpu_native.py "mpirun -np 2 /path/to/spdyn"

Developer notes
=======================================================================

Architecture
-----------------------------------------------------------------------

The Fortran side converts GENESIS's arrays (``s_domain``, ``s_enefunc``,
``s_constraints``, ``s_boundary``) into plain C descriptors declared in
``gpu_core_abi.h``; ``sp_gpu_core_abi.fpp`` mirrors that header one for
one, so a change to either is a change to both.  The device then owns the
state: one slot per owned atom, atoms grouped by rigid group, groups sorted
by cell.  Each rank owns the cells of its domain plus ghost cells.  Particle
data crosses to the host only at output, restart and replica-exchange
boundaries.

A step is driven from ``sp_gpu_core_step.fpp``, which is dispatched from
``sp_md_vverlet.fpp`` and ``sp_md_respa.fpp``.  The host decides (list
rebuild, output, barostat scalars); the device does the work: pair list
(rebuilt every ``nbupdate_period`` steps), real-space and bonded forces,
the reciprocal sum, SETTLE/SHAKE/RATTLE, the integrator and the thermostat.
Forces, energies and virials are summed in 64-bit fixed point
(``gpu_fixed_sum.cuh``), so results do not depend on thread scheduling.
Eligibility is one collective decision at setup; a decline falls back to
the standard path on all ranks with the reason logged.  The binary reads no
environment variable.

File map (``src/spdyn``)
-----------------------------------------------------------------------

``sp_gpu_core.fpp``, ``sp_gpu_core_abi.fpp``, ``sp_gpu_core_step.fpp``
  Fortran bridge: descriptor conversion, ABI mirror, step loop and
  fallback decision.
``gpu_core.cpp``, ``gpu_core_abi.h``, ``gpu_core_internal.h``
  Host context, grouped state import/export, canonical term records.
``gpu_core_native.h``, ``gpu_rebuild.cu``
  Device state; the rebuild (group sort, cell runs, pair list, masks).
``gpu_force.cu``, ``gpu_nbcluster.cu``, ``gpu_nbcluster_fit.h``
  Real-space and bonded kernels; the cluster-pair kernel of MIXED, whose
  design follows the GROMACS GPU kernels as published.
``gpu_step.cu``, ``gpu_npt.cu``, ``gpu_remd.cu``
  Integrator, constraints, thermostat; MTK barostat; REMD boundary.
``gpu_pme.cu``, ``gpu_pme_recip*.{cu,hpp}``, ``gpu_pencil_fft.*``,
``gpu_layout_schedule.hpp``, ``gpu_mesh_exchange_*``
  PME: spread, solve, gather; pencil FFT (cuFFT) and the mesh transposes.
``gpu_domain.cu``, ``gpu_migration.cu``
  Domain geometry, halo, force return, atom migration.
``gpu_xchg.cu``, ``gpu_core_xchg.h``
  The one exchange primitive (peer copies, CUDA-aware MPI, staging, the
  transport thread) used by every exchange above.

Runtime rules
-----------------------------------------------------------------------

There are only two places where the run chooses a path at run time.

Fused or host-driven exchange
  A node-local halo or mesh exchange is either fused into the step's
  stream (copy engines between device signal words) or driven from the
  host.  The choice is a fixed topology rule (``ce_admission`` in
  ``gpu_step.cu``): host-driven when every peer link of the run is PCIe and
  some GPU exchanges with two or more peers, fused otherwise.  Both paths
  move the same bytes into the same slots, so results are identical.

Peer or staged path
  For an edge between two GPUs of a node with a peer link, ``peer_path`` in
  ``gpu_xchg.cu`` chooses between staging through pinned host memory,
  peer copy engines and peer SM stores.  It is the one timing-based
  decision: it reads the start-up probe's measurements (every rank moving to
  all its node peers beside compute) because no fixed rule reproduces it
  across PCIe and NVLink machines.  The elected routes are printed with
  each plan (``Native_Xchg> plan=``).  This choice moves bytes differently,
  never different bytes.

Everything else (kernel variants, launch shapes, cadences) is fixed by the
input and the geometry.  ``tests/regression_test/test_gpu_native.py``
checks that the native core is engaged.
