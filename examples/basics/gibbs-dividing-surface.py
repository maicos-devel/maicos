#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Gibbs Dividing Surface
======================

This example shows how to construct the Gibbs dividing surface (GDS) of a planar
liquid film from a density profile. The GDS is the simplest box model of an
interface: it replaces the smooth profile by a step that jumps between two bulk
values at a single plane, placed such that the surface excess vanishes. For the
underlying theory see the explanations on :ref:`box-models`.

We start from a precomputed mass-density profile of water confined in a graphene
slit. The profile was obtained with :class:`maicos.DensityPlanar` and stored for
later use, exactly as for the dielectric box model in the
:ref:`userdoc-how-to-dielectrics` section.

"""  # noqa: D415

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# It is good practice to propagate the uncertainty of the profile to the
# dividing surface. We therefore define a trapezoidal integral that also returns
# an error estimate, and a helper that performs the Gibbs construction for a
# single interface.


def trapz_err(vals, errs, dx):
    """Trapezoidal integral of a profile with propagated uncertainty."""
    integral = dx / 2 * np.sum(vals[1:] + vals[:-1])
    integral_err = dx / 2 * np.sum(errs[1:] + errs[:-1])
    return integral, integral_err


def gibbs_surface(z, rho, err, rho_liq, rho_liq_err, z0=0.0, side="+"):
    r"""Equimolar dividing surface of one interface relative to the plane ``z0``.

    The exterior (vapour) density is taken to be zero, so the surface-excess
    condition reduces to :math:`\rho_\mathrm{liq}\,(z_\mathrm{G} - z_0) =
    \int \rho\,\mathrm{d}z` over the corresponding half of the profile.
    """
    mask = z >= z0 if side == "+" else z <= z0
    dx = z[1] - z[0]
    coverage, coverage_err = trapz_err(rho[mask], err[mask], dx)
    sign = 1.0 if side == "+" else -1.0
    z_g = z0 + sign * coverage / rho_liq
    z_g_err = np.sqrt(
        (coverage_err / rho_liq) ** 2 + (coverage * rho_liq_err / rho_liq**2) ** 2
    )
    return z_g, z_g_err


# %%
# .. rubric:: Load the precomputed density profile
#
# The file stores the planar mass-density profile of the confined water,
# centred on the water slab. The three columns are the bin position, the mass
# density and its uncertainty.
data = np.loadtxt("graphene_water_density.dat")
z, rho, rho_err = data[:, 0], data[:, 1], data[:, 2]

# %%
# .. rubric:: Reference densities
#
# The Gibbs construction needs the two bulk values that the step model jumps
# between. On the outside of the film the water density vanishes, so the vapour
# reference is zero. For the liquid reference we average the profile over the
# centre of the film, weighting each bin by its inverse squared uncertainty.
central = np.abs(z) < 1.5
weights = 1 / rho_err[central] ** 2
rho_liq = np.average(rho[central], weights=weights)
rho_liq_err = np.sqrt(1 / np.sum(weights))
print(f"Liquid reference density: {rho_liq:.3f} ± {rho_liq_err:.3f} u/Å³")

# %%
# The film is thin and strongly layered, so it never fully recovers a flat bulk
# plateau. The dividing surface therefore depends on how the liquid reference is
# chosen; this caveat for confined systems is discussed in the
# :ref:`box-models` explanations. :footcite:p:`schlaichWaterDielectricEffects2016`

# %%
# .. rubric:: Construct the Gibbs dividing surfaces
#
# The film has two symmetric interfaces. We place a dividing surface on each
# side relative to the slab centre at :math:`z = 0`.
z_g_plus, z_g_plus_err = gibbs_surface(z, rho, rho_err, rho_liq, rho_liq_err, side="+")
z_g_minus, z_g_minus_err = gibbs_surface(z, rho, rho_err, rho_liq, rho_liq_err, side="-")
thickness = z_g_plus - z_g_minus
thickness_err = np.sqrt(z_g_plus_err**2 + z_g_minus_err**2)

print(f"Gibbs surface (right): {z_g_plus:.2f} ± {z_g_plus_err:.2f} Å")
print(f"Gibbs surface (left):  {z_g_minus:.2f} ± {z_g_minus_err:.2f} Å")
print(f"Equimolar film thickness: {thickness:.2f} ± {thickness_err:.2f} Å")

# %%
# .. rubric:: Plot the profile together with the box model
#
# The box model is the step function that is constant at ``rho_liq`` between the
# two dividing surfaces and zero outside. By construction it encloses the same
# area (the same amount of water) as the real profile.
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(z, rho, label="density profile")
ax.axhline(rho_liq, color="black", linestyle="dashed", linewidth=1, label="liquid bulk")
ax.plot(
    [z.min(), z_g_minus, z_g_minus, z_g_plus, z_g_plus, z.max()],
    [0, 0, rho_liq, rho_liq, 0, 0],
    color="C1",
    label="box model",
)
for z_g in (z_g_minus, z_g_plus):
    ax.axvline(z_g, color="C1", linestyle=":", linewidth=1)

ax.set_xlabel(r"$z$ [$\AA$]")
ax.set_ylabel(r"$\rho$ [u $\AA^{-3}$]")
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %%
# References
# ----------
# .. footbibliography::
