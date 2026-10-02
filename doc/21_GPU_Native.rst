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

.. _gpu_native_architecture:

Architecture at a glance
=======================================================================

.. figure:: images/gpu-native-architecture.png
   :alt: Native GPU architecture: host setup and control, persistent GPU state and kernels, and communication between MPI ranks.
   :width: 100%
   :target: images/gpu-native-architecture.svg

   One MPI rank drives one GPU. GENESIS keeps setup, control and output on
   the host; the native core owns the MD state and computations on the
   device. The boxes group responsibilities rather than timestep order.

Particle data is exported at trajectory, restart and replica-exchange
boundaries. Communication between ranks may stage through host memory
when a direct GPU route is unavailable; the MD computations remain on
the GPU. See the developer architecture and file map below for the
implementation details.

.. _gpu_native_quick_start:

Quick start
=======================================================================

Use an NVIDIA GPU with a CUDA toolkit containing ``nvcc`` and cuFFT,
compatible C/C++ and Fortran compilers, MPI with C and Fortran support,
BLAS/LAPACK, GNU Make, Autoconf and Automake. On a cluster, load the site's
compiler, MPI and CUDA modules first. See :ref:`getting_started` for
compiler and library options.

1. Clone and build the GPU-enabled, double-precision binary::

     git clone --recurse-submodules https://github.com/wmarti/genesis-native.git
     cd genesis-native
     autoreconf -i
     ./configure --enable-gpu --enable-double --with-gpuarch=sm_80 \
       --prefix="$PWD" CC=mpicc FC=mpif90
     make -j4
     make install

   Replace ``sm_80`` with your target GPU's CUDA architecture. If CUDA is
   outside the compiler's search paths, add ``--with-cuda=/path/to/cuda``.
   The installed executable is ``bin/spdyn``. Both native precision modes
   use this double-precision build.

2. Start from a complete GENESIS SPDYN input file for your system. Add the
   native keyword to its existing [DYNAMICS] section::

     [DYNAMICS]
     gpu_resident = YES

   Full double precision is the default. To use Mixed precision with PME,
   add this to the existing [ENERGY] section of the same input::

     [ENERGY]
     nonbond_precision = MIXED

   These are additions to a complete input, not a full simulation deck.
   GROMACS users should use the GENESIS input format and the supported
   GROAMBER topology and coordinate inputs described in :ref:`input`.
   Check the supported simulations below before enabling the core.

3. Run from your simulation directory with one MPI rank per GPU::

     mpirun -np 1 /absolute/path/to/genesis-native/bin/spdyn md.inp > md.log

   Replace the executable and input paths. For four allocated GPUs use
   four ranks, with your site's launcher and binding rules assigning a
   distinct GPU to each rank.

4. Check rank 0's log for native execution::

     Setup_GPU_Core> requested=YES effective=native precision=FP64 reason=eligible

   Mixed runs report ``precision=mixed``. If the line says
   ``effective=cpu``, read its ``reason`` and the fallback rules below.
   Native-only settings, including Mixed precision, stop with the reason
   when the requested simulation cannot use the native core.

Example output
=======================================================================

These excerpts are from recorded native-core validation runs. Timing,
repeatability and agreement with a reference are separate checks.

B200: actual timing report (Mixed precision)
-----------------------------------------------------------------------

The STMV tuned deck ran 3000 steps at 2 fs on 1, 2 and 4 B200 GPUs.
Each rank count has six legs: three interleaved repetitions of two arms
of the same binary. Here are the six recorded 4-GPU legs::

  np  arm  rep  rc  ms_step  e
  4  A  1  0  1.1746  c3989b7c6ae67d99e896c3a18a246321
  4  C  1  0  1.1744  c3989b7c6ae67d99e896c3a18a246321
  4  C  2  0  1.1747  c3989b7c6ae67d99e896c3a18a246321
  4  A  2  0  1.1740  c3989b7c6ae67d99e896c3a18a246321
  4  A  3  0  1.1745  c3989b7c6ae67d99e896c3a18a246321
  4  C  3  0  1.1735  c3989b7c6ae67d99e896c3a18a246321

The recorded comparison and audit include::

  np4 step 500: max rel diff 0.00e+00 (TIME), total energy -2824982.647100 vs -2824982.647100
  AUDIT PASS legs=18
  DONE audit_rc=0

How to read the columns:

* ``np``: MPI rank count, one GPU per rank.
* ``arm`` and ``rep``: interleaved A/C arms and repetition number. Both
  arms here ran the same binary.
* ``rc``: process return code; zero means the process completed.
* ``ms_step``: steady-state wall-clock milliseconds per MD step.
* ``e``: MD5 of the INFO-row output with the external wall-clock prefix
  removed. Matching values within one rank count check repeatability of
  that output; they do not prove agreement with an FP64 reference.
* ``AUDIT PASS``: all 18 legs passed the run audit, including native
  engagement, the full INFO series and a valid timer.

The published medians are:

.. list-table:: B200, STMV tuned deck, Mixed precision
   :header-rows: 1
   :widths: 15 25 25 35

   * - GPUs
     - Median ms/step
     - ns/day at 2 fs
     - Valid legs
   * - 1
     - 3.3638
     - 51.4
     - 6
   * - 2
     - 2.0220
     - 85.5
     - 6
   * - 4
     - 1.1745
     - 147.1
     - 6

The timing harness prefixes each INFO row with a wall-clock timestamp.
It takes the slope from one third of the run to the last INFO row before
the terminal step (which includes teardown)::

  ms_per_step = 1000 * (wall_time_last - wall_time_first)
                     / (step_last - step_first)

This is elapsed wall time per MD step, not the simulated time column.
The ns/day conversion also depends on the physical timestep, so compare
ms/step when two systems use different timesteps.

`Read the complete 18-leg timing excerpt <examples/gpu-native-b200-stmv-mixed.txt>`_.
Paths, host identifiers and scheduler metadata have been omitted; the
measurement rows and check results are unchanged. The release's
`timing table <https://github.com/wmarti/genesis-native/releases/download/gpu-resident-v1/timing-gpu-resident-v1.tsv>`_
contains all measured systems, GPU models, precisions and settings.

Blackwell: actual FP64 repeatability output
-----------------------------------------------------------------------

An NVE repeatability check on RTX PRO 6000 Blackwell reported::

  ENS nve np1 FP64_BITWISE PASS  native A=native C=native
  ENS nve np2 FP64_BITWISE PASS  native A=native C=native
  ENS nve np4 FP64_BITWISE PASS  native A=native C=native

``native A=native C=native`` confirms both runs used the native core.
``FP64_BITWISE PASS`` compares repeated FP64 outputs for the same rank
count. The corresponding NVE audit passed six legs. Reproducibility is
defined for a fixed rank count; energy digests need not match between
different rank counts.

`Read the FP64 NVE excerpt <examples/gpu-native-blackwell-fp64-nve.txt>`_.
This excerpt is limited to NVE; it does not stand for every ensemble or
for Mixed-versus-FP64 accuracy.

Mixed precision: checking agreement with FP64
-----------------------------------------------------------------------

The separate released suite compared Mixed runs with fine-table FP64
references (``table_density = 2000``), using the upstream relative
``tolerance_single = 3.0e-5``. The reported largest differences were
``1.2e-5`` for van der Waals energy and ``1.8e-6`` for total energy.
Those are validation summaries from the RTX 4090 and RTX 3090 suite,
not stdout from the B200 timing run. At 1 and 2 ranks, 48/56 and 46/56
Mixed cases completed with zero failed comparisons; the remaining cases
declined the native core and stopped.

To check your own GPU build with the included native regression cases::

  cd tests/regression_test
  python3 test_gpu_native.py "mpirun -np 2 /absolute/path/to/genesis-native/bin/spdyn"

Use two allocated GPUs for this example. The driver verifies native
engagement and compares energies with the supplied references; it also
checks cases that are expected to stop. The driver uses ``1.0e-6`` for
double-precision energy comparisons and ``3.0e-5`` for Mixed, with its
documented virial allowance. Read its final passed/failed/aborted counts
alongside the per-case output.

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

  $ cd tests/regression_test
  $ python3 test_gpu_native.py "mpirun -np 2 /absolute/path/to/genesis-native/bin/spdyn"

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
