.. _dielectric-explanations:

===============================
Dielectric constant measurement
===============================

Dielectric Response of Homogeneous, Isotropic Fluids
====================================================
The linear dielectric response of a material relates the displacement field :math:`D` to
the electric field :math:`E`, which in the isotropic, homogenous case can be written as
(in SI units)

.. math::
   \mathbf{D} = \varepsilon_0 \varepsilon \mathbf{E}

where :math:`\varepsilon_0` is the vacuum permittivity, and :math:`\varepsilon` is the
dielectric constant of the insulating medium.

One can relate the dielectric constant of a material to the fluctuations of the dipole
moment of a sub-sample even without any perturbation by an external field. Relations of
this sort have been known since the 1930s and follow from the fluctuation-dissipation
theory :footcite:p:`kirkwoodDielectricPolarizationPolar1939`. Depending on the boundary
conditions, this equation takes different forms, however, the most common boundary
conditions of molecular dynamics simulations are tin-foil boundary conditions in
conjunction with an Ewald-summation type approach. In this case, we get for a bulk
material

.. math::
   \varepsilon = 1 + \frac{\langle M^2 \rangle - \langle M \rangle^2}
                          {3 \varepsilon_0 V k _{\mathrm{B}} T}

where :math:`M` is the dipole moment of the sample, :math:`V` is its volume, :math:`k
_\mathrm{B}` is the Boltzmann constant and :math:`T` is the temperature.

The dipole moment is defined by

.. math::
   M = \sum_i \mathbf{r}_i q_i

where :math:`\mathbf{r}_i` is the position of the :math:`i`-th particle and :math:`q_i`
is the charge of the :math:`i`-th particle. Notably, this allows the calculation of the
dielectric response from equilibrium simulations without the need to explicitly define
an external field in simulations.

This analysis - valid for isotropic and homogeneous systems - is implemented in
:class:`MDAnalysis.analysis.dielectric.DielectricConstant` and can directly be applied
to trajectories of homogeneous systems.

Dielectric Response of Fluids at Interfaces and in Confinement
==============================================================

Electrostatic Theory
--------------------
The relationship between the electric field and the dielectric response shown above is
only valid for isotropic homogeneous systems, where the properties of the material are
the same throughout. However, there is also a need for calculating the dielectric
response of anistropic inhomogeneous systems. For instance, fluids confined in a porous
material are of great importance for many technological processes, such as energy
storage devices like batteries and capacitors. In these devices, a nano-porous electrode
is used to increase the surface area and improve the capacity of the device. Another
common example are catalysts, where an increased surface area is used to increase the
rate of a chemical reaction, and thus porous catalysts are often utilized.

The presence of interfaces alters the dielectric response of the fluid in two ways.
First, the response is not isotropic anymore, but depends on the orientation of the
electric field. Second, the response varies with the distance from the surface of the
porous material, i.e., it becomes inhomogeneous.

In the following discussion, we will focus on pores with planar symmetry, also known as
"slit pores" and implemented in :class:`maicos.DielectricPlanar`. However, similar
concepts apply to other types of pore geometries, such as ones with cylindrical or
spherical symmetries implemented in :class:`maicos.DielectricCylinder` and
:class:`maicos.DielectricSphere`.

Without loss of generality, we will assume that the pore is aligned along the
:math:`z`-axis.

The non-local, anisotropic, linear dielectric response of a fluid can generally be
written as :footcite:p:`bonthuisProfileStaticPermittivity2012`

.. math::
   D(\mathbf{r}) = \varepsilon_0 \int_V \mathrm{d}^3 r'
                \varepsilon(\mathbf{r}, \mathbf{r}') E(\mathbf{r}')

where :math:`\varepsilon(\mathbf{r}, \mathbf{r}')` is the dielectric tensor, which
describes how the dielectric response of the fluid at position :math:`\mathbf{r}` is
affected by the electric field :math:`E(\mathbf{r}')` throughout the volume :math:`V` of
the fluid. The convolution integral accounts for the non-local influences of the fluid
response at other locations.

In planar symmetry, we can simplify the above expression further, because the Maxwell
relations give

.. math::
     \nabla \times \mathbf{E} = 0

in the absence of external magnetic fields. Because of the planar symmetry, we know that
the :math:`\mathbf{E}` only varies with respect to :math:`z`. Hence, the above gives
:math:`\partial_z E_y = \partial_z E_x = 0`, implying that the parallel components of
the electric field do not vary with :math:`z`.

Thus, we can simplify the anisotropic, non-linear equation above in the parallel case to

.. math::
     D _\parallel = \epsilon_0 E_\parallel \int \mathrm{d}z'
     \epsilon_\parallel(z, z') =: \epsilon_0 \epsilon_\parallel(z) E_\parallel

where the marginal integration of :math:`\varepsilon_\parallel (\mathbf{r},
\mathbf{r}')` defines the dielectric profile :math:`\varepsilon_\parallel(z)`. It is
important to note that this derivation starts with non-local assumptions and is exact in
the case of planar geometries discussed here (similar derivations apply also for
cylindrical and spherical symmertries). Thus, :math:`\varepsilon_\parallel(z)` fully
captures the non-locality of the confined fluid's response and does not require
additional assumptions.

In the absence of "free charges" we can use the macroscopic Maxwell equation

.. math::
     \nabla \cdot \mathbf{D} = 0

to derive the perpendicular dielectric profile.

.. warning::
    This requires that no free charges are used in simulations, which
    means that no ions can be included in simulations. This is a common pitfall
    and leads to a wrong analysis.

The above equation gives us the important relation of :math:`\partial_z \mathbf{D}_z =
0`, which implies that the perpendicular components of the displacement field do not
vary with :math:`z`. Thus, if we start with the inverse dielectric response, defined as

.. math::
     E(z) = \varepsilon_0^{-1} \int \mathrm{d} z' \varepsilon^{-1}(z, z') D(z')

where :math:`\varepsilon^{-1}(z, z')` is the matrix inverse of the dielectric tensor.
Similar to above, we use the fact that :math:`D` does not vary with :math:`z` and
simplify

.. math::
     E_\perp = \epsilon_0^{-1} D_\perp \int \mathrm{d}z'
               \epsilon_\perp^{-1}(z, z') =: \epsilon_0^{-1}
               \epsilon_\perp^{-1}(z)  D_\perp

where the marginal integration of :math:`\varepsilon_\perp^{-1} (\mathbf{r},
\mathbf{r}')` defines the inverse dielectric profile :math:`\varepsilon_\perp^{-1}(z)`.

**In summary**, if one has no magnetic fields and no free charges, the dielectric
profiles :math:`\varepsilon^{-1}_\bot (z)` and :math:`\varepsilon_\parallel(z)` fully
define the linear, anisotropic, non-local response of a system in planar confinement.

Fluctuation-Dissipation Theorem
-------------------------------
As was briefly discussed for the homogenous case, the dielectric response of a system
can be calculated from equilibrium simulations without the need to explicitly define an
external field in simulations, using a fluctuation dissipation theorem. This can be
derived by identifying the linear response under consideration, in this case the
dielectric response to a derivative of the expected value of an observable, in this case
the polarization density. The expectation value is calculated using statistical
mechanics. One can then show :footcite:p:`sternCalculationDielectricPermittivity2003`
:footcite:p:`bonthuisProfileStaticPermittivity2012`
:footcite:p:`schlaichWaterDielectricEffects2016` that the dielectric response formalism
is given by

.. math::
     \epsilon_\parallel(z) = 1 + \frac{\langle m_\parallel(z) M_\parallel \rangle
                            - \langle m_\parallel (z) \rangle \langle M \rangle}
                            {\epsilon_0 k_\mathrm{B}T}

for the **parallel** dielectric profile, and

.. math::
     \epsilon_\perp^{-1}(z) = 1 - \frac{\langle m_\perp(z) M_\perp \rangle
                             - \langle m_\perp (z) \rangle \langle M \rangle}
                             {\epsilon_0 k_\mathrm{B}T},

for the **inverse** perpendicular dielectric profile.

Note that we still need to define how to calculate :math:`m_\parallel(z)` and
:math:`m_\perp(z)`. For the perpendicular polarization density, we have
:footcite:p:`bonthuisProfileStaticPermittivity2012`

.. math::
     m_\perp (z) = - \int^z _0 \mathrm{d}z' \rho(z').

For the parallel case, we have to derive the lateral component of the polarization
density as a function of the coordinate :math:`z`. This can be done by introducing
multiple virtual cuts perpendicular to any lateral axis, such as the :math:`x` or
:math:`y` axis :footcite:p:`bonthuisProfileStaticPermittivity2012`
:footcite:p:`schlaichWaterDielectricEffects2016`. During this step one has to take care
to only cut molecules along this cutting plane, which requires careful treatment of the
periodic boundary conditions commonly employed in simulations. Identifying the
(non-zero) total charge on one side of the cut with the surface charge along the plane
of the virtual cut via Gauss' theorem we can integrate out the dependency of the lateral
axis of the cut and average over multiple such cuts. This gives a good estimate for the
average surface charge density :math:`\sigma(z)` w.r.t the coordinate :math:`z`.
Finally, we can identify

.. math::
     m_\parallel (z) = \mp \sigma (z).

Boundary Conditions
-------------------
The above equations for :math:`\varepsilon _\parallel (z)` and :math:`\varepsilon
_\perp^{-1} (z)` are derived under 2d periodicity. In simulations, this entails using
periodic boundary conditions only in the :math:`x` and :math:`y` directions. In most of
the typically employed simulation codes, electrostatics are calculated using a
Ewald-summation type approach. This includes direct Ewald sums or the faster meshed
Ewald sums (such as P3M, and PME). However, in their usual formulation these codes
calculate 3d-periodic systems and thus do not meet the assumptions of the derivation
shown above.

In order to use the above, one can use the 2d Ewald sum or corrections thereof, such as
the correction of Yeh and Berkovitz :footcite:p:`yehEwaldSummationSystems1999` or the
ELC :footcite:p:`arnoldElectrostaticsPeriodicSlab2002`.

However, one can also correct for the 3d electrostatics of an uncorrected Ewald-sum in
the fluctuation dissipation formalism directly as shown in refs.
:footcite:p:`sternCalculationDielectricPermittivity2003`
:footcite:p:`schlaichWaterDielectricEffects2016`

For tin-foil boundary conditions  (which are most commonly used), one gets
:footcite:p:`schlaichWaterDielectricEffects2016`

.. math::
         \epsilon_\perp^{-1} (z) = 1 - \frac{\langle m_\perp(z) M_\perp\rangle
         - \langle m_\perp(z)\rangle \langle M_\perp \rangle}{\epsilon_0
         k_{\text{B}}T + C_\perp/V},

where :math:`C_\perp = \int \mathrm{d} m_\perp(z)`.

A very close formula :footcite:p:`sternCalculationDielectricPermittivity2003` can also
be derived for vacuum boundary conditions at infinity, which some simulation codes also
provide.

.. note::
    The above equation reduces to the correct 2d periodic system if one
    includes an explicit vacuum layer in the :math:`z` direction of infinite
    (sufficiently large) size, such that the influence between periodic images
    over the :math:`z` direction can be approximated as a dipole interaction.
    This approach is analogous to the Yeh and Berkovitz correction
    :footcite:p:`yehEwaldSummationSystems1999` and
    may be used to calculate the dielectric profiles for physical systems with
    2d-symmetry when corrections are not available. In these situations, we
    recommend to use a padding vacuum layer such that the system is 3x the
    physical system size in :math:`z` direction.

    However, there are systems which truly are 3d-periodic, such as stacks of lipid
    membranes. In these cases, one has to also use the above formula which includes the
    dipole corrections, but only simulate the physical system, without a padding vacuum
    layer.

These boundary conditions can be specified using the parameters ``is_3d``. If a
simulation used vacuum boundary conditions at infinity, one can use the ``vac``
parameter.

References
----------
.. footbibliography::
