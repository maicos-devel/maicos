.. _box-models:

===========================================
Box models for interfacial and slab systems
===========================================

Many quantities computed by MAICoS — density, dielectric permittivity, velocity — vary
smoothly across an interface. For thermodynamic and continuum-mechanical bookkeeping it
is often convenient to compress such a profile into a piecewise-constant *box model*:
a constant bulk value on each side of an idealized, infinitely thin dividing surface.
The position of that surface is not unique; it depends on which quantity is being
conserved across the discontinuity. This page collects the conventions MAICoS users
most commonly encounter and shows how to obtain each of them from the analysis output.

The discussion below is written for planar geometry, where the dividing surface
:math:`z_\mathrm{d}` is a plane perpendicular to the analysis axis. The same ideas
apply to cylindrical and spherical symmetries with the appropriate Jacobians; see
:class:`maicos.DensityCylinder`, :class:`maicos.DielectricCylinder` and the analogous
spherical classes.

-------------------------
Gibbs dividing surface
-------------------------

The Gibbs dividing surface (GDS) is the classical construction from Gibbs' treatment of
heterogeneous systems. Given a density profile :math:`\rho(z)` that interpolates
between two bulk values :math:`\rho_\mathrm{a}` (for :math:`z\to-\infty`) and
:math:`\rho_\mathrm{b}` (for :math:`z\to+\infty`), the GDS is the plane
:math:`z_\mathrm{G}` at which the surface excess vanishes:

.. math::
    \Gamma(z_\mathrm{G}) \;=\; \int_{-\infty}^{z_\mathrm{G}}
        \bigl[\rho(z) - \rho_\mathrm{a}\bigr]\,\mathrm{d}z
    \;+\; \int_{z_\mathrm{G}}^{+\infty}
        \bigl[\rho(z) - \rho_\mathrm{b}\bigr]\,\mathrm{d}z
    \;\stackrel{!}{=}\; 0.

Equivalently, :math:`z_\mathrm{G}` is the unique position that makes the integral of
the step model

.. math::
    \rho_\mathrm{step}(z) \;=\;
    \begin{cases}
        \rho_\mathrm{a}, & z < z_\mathrm{G}, \\
        \rho_\mathrm{b}, & z > z_\mathrm{G},
    \end{cases}

equal to that of the true profile :math:`\rho(z)`. On a finite interval
:math:`[z_0, z_1]` straddling the interface this gives the explicit solution

.. math::
    z_\mathrm{G} \;=\; z_1 \;-\;
    \frac{1}{\rho_\mathrm{b}-\rho_\mathrm{a}}
    \int_{z_0}^{z_1} \bigl[\rho(z)-\rho_\mathrm{a}\bigr]\,\mathrm{d}z.

For a liquid–vapour interface (:math:`\rho_\mathrm{b}=0`) the GDS reduces to the
equimolar (or equimass) surface, i.e. the plane that places half of the missing
density on either side.

The choice of which density enters :math:`\rho(z)` matters. The GDS of the solvent
mass density is in general not the GDS of the solute number density, nor of the charge
density; in heterogeneous mixtures one has to be specific about which species defines
the dividing surface. For an aqueous interface the convention is usually the oxygen
number density or the total water mass density.

.. code-block:: python

    import MDAnalysis as mda
    import numpy as np
    import maicos

    u = mda.Universe("topol.tpr", "traj.xtc")
    water = u.select_atoms("resname SOL")

    dens = maicos.DensityPlanar(water, dens="mass", bin_width=0.1, refgroup=water)
    dens.run()

    z = dens.results.bin_pos
    rho = dens.results.profile
    bw = z[1] - z[0]

    # bulk values from regions away from the interface
    rho_a = rho[: len(rho) // 8].mean()
    rho_b = rho[-len(rho) // 8 :].mean()

    # direct evaluation of the formula above
    z_gds = z[-1] - np.sum(rho - rho_a) * bw / (rho_b - rho_a)

When the system is symmetric around the reference group (e.g. a slab between two
identical interfaces), it is usually more robust to set ``sym=True`` on
:class:`maicos.DensityPlanar` and to determine the GDS on one half only.

----------------------------
Dielectric dividing surface
----------------------------

The dielectric dividing surface (DDS) is the analogue of the GDS for the dielectric
profile :math:`\varepsilon(z)`. It is the plane that bounds the effective
bulk-dielectric slab whose integrated response matches the integrated response of
the real profile. For a symmetric slab centred at :math:`z=0` with perpendicular
inverse dielectric profile :math:`\varepsilon_\perp^{-1}(z)` and bulk value
:math:`\varepsilon_\perp^{-1,\mathrm{bulk}}` in the centre, the effective slab
thickness :math:`h_\mathrm{eff}` is defined by

.. math::
    \int_{-\infty}^{+\infty}
        \bigl[1 - \varepsilon_\perp^{-1}(z)\bigr]\,\mathrm{d}z
    \;=\;
    h_\mathrm{eff}\,\bigl(1 - \varepsilon_\perp^{-1,\mathrm{bulk}}\bigr),

and the perpendicular DDS positions sit at :math:`z_\mathrm{D}^\pm = \pm
h_\mathrm{eff}/2`. The same construction applied to :math:`\varepsilon_\parallel(z)`
gives a parallel DDS, which in general differs from the perpendicular one. The shift
:math:`z_\mathrm{D} - z_\mathrm{G}` quantifies the dielectric *dead layer* near the
interface — the region in which the fluid is structurally present but contributes
less than the bulk to the dielectric response.

The DDS construction has been used to map molecular-dynamics dielectric profiles onto
continuum slab models in :footcite:t:`bonthuisProfileStaticPermittivity2012` and
:footcite:t:`schlaichWaterDielectricEffects2016`. It is less standard than the GDS
and the precise definition of "bulk" matters: in confined systems the bulk reference
should be taken from a region of the profile that has recovered its homogeneous
value, which is not always available for narrow pores.

.. code-block:: python

    import maicos

    eps = maicos.DielectricPlanar(
        water, bin_width=0.5, refgroup=water, sym=True, is_3d=True,
    )
    eps.run()

    z = eps.results.bin_pos
    eps_perp_inv = 1.0 + eps.results.eps_perp  # MAICoS returns the reduced profile
    bw = z[1] - z[0]

    # bulk value from the slab centre
    centre = np.argmin(np.abs(z))
    eps_perp_inv_bulk = eps_perp_inv[centre - 2 : centre + 3].mean()

    # equation above, solved for h_eff
    h_eff = np.sum(1.0 - eps_perp_inv) * bw / (1.0 - eps_perp_inv_bulk)
    z_dds = h_eff / 2.0  # symmetric slab, positive interface

The shift :math:`z_\mathrm{D} - z_\mathrm{G}` is then ``z_dds`` minus the Gibbs
surface obtained from the mass-density profile of the same trajectory.

--------------------------------------------
Hydrodynamic boundary and slip-length models
--------------------------------------------

For systems under shear or pressure-driven flow the natural box model is the
hydrodynamic one. The Navier slip boundary condition relates the fluid velocity
at the wall to its shear rate via the slip length :math:`b`:

.. math::
    u(z_\mathrm{H})\;=\; b\,\left.\frac{\mathrm{d}u}{\mathrm{d}z}\right|_{z_\mathrm{H}},

where :math:`z_\mathrm{H}` is the hydrodynamic boundary position. Operationally,
:math:`z_\mathrm{H}` and :math:`b` are obtained by fitting the bulk-form velocity
profile — linear in Couette flow, parabolic in Poiseuille flow — to the central,
fully developed region of the measured profile and extrapolating to the plane at
which :math:`u(z)` would vanish (for stationary walls). That extrapolated plane
defines :math:`z_\mathrm{H}`, and :math:`b` is its signed distance to the physical
wall — positive when the no-shear plane lies outside the fluid, negative when it
lies inside.

In general :math:`z_\mathrm{H}` does not coincide with the Gibbs surface of the
solvent density. Reporting both is useful: the difference
:math:`z_\mathrm{H} - z_\mathrm{G}` is a measure of the depletion or stagnation
layer near the wall and varies strongly with surface chemistry.

.. code-block:: python

    import numpy as np
    import maicos

    vel = maicos.VelocityPlanar(
        water, vdim=0, dim=2, bin_width=0.5, refgroup=water,
    )
    vel.run()

    z = vel.results.bin_pos
    u = vel.results.profile

    # fit a parabola u(z) = a (z - z_c)**2 + u_max in the central, fully developed region
    mask = np.abs(z) < 0.5 * z.max()
    a, b_lin, c = np.polyfit(z[mask], u[mask], deg=2)

    # planes where the parabolic continuation crosses zero
    z_pm = np.roots([a, b_lin, c])
    z_hydro = z_pm[z_pm > 0].min()  # right-hand boundary

    # slip length: distance from the hydrodynamic boundary to the physical wall
    z_wall = ...  # e.g. carbon-layer position from a density profile
    slip_length = z_wall - z_hydro

For Couette flow replace the quadratic fit with a linear one and extract the
intercepts the same way.

----------------------
When to use which box
----------------------

A few rules of thumb:

* If you want a thermodynamic accounting of an interface (surface tension, adsorption,
  Gibbs free energy) use the **Gibbs surface** of the relevant species.
* If you want to map a polarization profile onto an effective continuum slab — for
  example to predict an image-charge interaction or a capacitance — use the
  **dielectric dividing surface** for the appropriate component.
* If you want to predict flow rates, pressure drops or apparent viscosities use the
  **hydrodynamic boundary** with the corresponding slip length. The Gibbs surface
  is generally a poor proxy for it.

These surfaces almost never coincide and they are not interchangeable. Reporting the
convention along with the numerical value is more important than the choice of
convention itself.

References
----------
.. footbibliography::
