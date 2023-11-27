# Mica data

A realistic mica-water interface.

The mica surface is modeled as two slabs of T-O-T layers, separated at the weak ionic
bonding layer. 587 water between the slabs.


.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Value

   * - Code
     - GROMACS
   * - Ensemble
     - NVT
   * - Force-field
     - water model = TIP4P/ɛ
   * - Timestep
     - 2 fs
   * - Frame number
     - 101
   * - Duration
     - 200 ps
   * - Dimensions
     - 2.9x3.0x4.0 nm**3
   * - Others
     - Temperature 300 K, h-bonds contrained by LINCS, coulombtype = PME
