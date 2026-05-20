"""
How-to: Effective-medium analysis for a single water model
===========================================================

This example shows the post-processing step for one water model. It starts from
precomputed planar dielectric profiles, extracts a bulk dielectric constant from
the center of the pore, and then estimates the effective-medium response.

Replace the ``base_path`` below with the files produced for your own simulation.
The parallel profile is expected in ``*_par.dat`` and the perpendicular profile in
``*_perp.dat``.

"""  # noqa: D415

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def draw_effective_box(ax, leff, value, color="C1", label=None):
    """Draw a finite effective-medium slab centered in the pore."""
    xmin, xmax = ax.get_xlim()
    x_center = 0.5 * (xmin + xmax)
    rect = Rectangle(
        (x_center - leff / 2, 0),
        leff,
        value,
        facecolor=color,
        edgecolor=color,
        linewidth=2,
        alpha=0.2,
        zorder=4,
        label=label,
    )
    ax.add_patch(rect)


# %%
def trapz_err(vals, errs, dx):
    """Integral of a profile with error estimation using the trapezoidal rule."""
    integral = dx / 2 * np.sum(vals[1:] + vals[:-1])
    integral_err = dx / 2 * np.sum(errs[1:] + errs[:-1])
    return integral, integral_err


def bulk_eps(z, eps, err, bulk_dist=15):
    """Estimate the bulk dielectric constant from the pore center."""
    bulk_filter = np.logical_and(z > bulk_dist, z < z[-1] - bulk_dist)

    if not np.any(bulk_filter):
        return np.nan, np.nan
    if not np.any(bulk_filter):
        return np.nan, np.nan

    weights = 1 / err[bulk_filter] ** 2
    eps_blk = np.average(eps[bulk_filter], weights=weights)
    eps_blk_err = np.sqrt(1 / np.sum(weights))
    weights = 1 / err[bulk_filter] ** 2
    eps_blk = np.average(eps[bulk_filter], weights=weights)
    eps_blk_err = np.sqrt(1 / np.sum(weights))

    return eps_blk, eps_blk_err


def calc_leff_trapz(z, profile, err, length, bulk_response):
    """Calculate the effective length from the profile using the trapezoidal rule."""
    dx = z[1] - z[0]
    integral, integral_err = trapz_err(profile, err, dx)
    leff = (integral - length) / (bulk_response - 1)
    leff_err = integral_err * np.abs(1 / (bulk_response - 1))
    return leff, leff_err


def effective_response(z, profile, err, leff, length, leff_err):
    """Effective-medium response from the profile and the effective length."""
    dx = z[1] - z[0]
    integral, integral_err = trapz_err(profile, err, dx)
    response = 1 + ((integral - length) / leff)
    response_err = integral_err * np.abs(1 / leff) + leff_err * np.abs(
        (integral - length) / leff**2
    )
    return response, response_err


def prepare_epsilon(base_path):
    """Load the parallel and perpendicular dielectric profiles from the given files."""
    par = np.loadtxt(f"{base_path}_par.dat")
    perp = np.loadtxt(f"{base_path}_perp.dat")

    z_par, eps_par = par[:, 0], par[:, 1] + 1
    z_par -= z_par[0]
    z_par, eps_par = par[:, 0], par[:, 1] + 1
    z_par -= z_par[0]

    z_perp, eps_perp_inv = perp[:, 0], perp[:, 1] + 1
    z_perp -= z_perp[0]
    z_perp, eps_perp_inv = perp[:, 0], perp[:, 1] + 1
    z_perp -= z_perp[0]

    return (z_par, eps_par, par[:, 2]), (z_perp, eps_perp_inv, perp[:, 2])


# %%
# ---------------------------------------------------------------------------
# Load one water-model example
# ---------------------------------------------------------------------------
# These files should contain the dielectric profile for one confined water
# system, for example SPC/E water in a slit pore.
base_path = "./tip4p_data/eps_l0d3"

(z_par, eps_par, eps_par_err), (z_perp, eps_perp_inv, eps_perp_err) = prepare_epsilon(
    base_path
)

# SPC/E bulk dielectric constant used as a reference value in this example.
eps_bulk = 75.0


# ---------------------------------------------------------------------------
# Parallel component
# ---------------------------------------------------------------------------
# %%
leff_par, leff_par_err = calc_leff_trapz(
    z_par, eps_par, eps_par_err, z_par[-1], eps_bulk
)
eps_eff_par, eps_eff_par_err = effective_response(
    z_par, eps_par, eps_par_err, leff_par, z_par[-1], leff_par_err
)


# ---------------------------------------------------------------------------
# Perpendicular component
# The perpendicular profile is stored as epsilon^{-1}(z).
# ---------------------------------------------------------------------------
# %%
leff_perp, leff_perp_err = calc_leff_trapz(
    z_perp, eps_perp_inv, eps_perp_err, z_perp[-1], 1 / eps_bulk
)
eps_eff_perp_inv, eps_eff_perp_inv_err = effective_response(
    z_perp, eps_perp_inv, eps_perp_err, leff_perp, z_perp[-1], leff_perp_err
)
print(
    f"Effective-medium estimate (parallel): {eps_eff_par:.2f} ± {eps_eff_par_err:.2f}"
)
perp_val = 1 / eps_eff_perp_inv
perp_err = eps_eff_perp_inv_err / eps_eff_perp_inv**2
print(f"Effective-medium estimate (perpendicular): {perp_val:.2f} ± {perp_err:.2f}")
print(f"Effective length (parallel): {leff_par:.2f} ± {leff_par_err:.2f}")
print(f"Effective length (perpendicular): {leff_perp:.2f} ± {leff_perp_err:.2f}")

# ---------------------------------------------------------------------------
# Bulk estimate from the pore center
# ---------------------------------------------------------------------------
eps_blk_par, eps_blk_par_err = bulk_eps(z_par, eps_par, eps_par_err)


# ---------------------------------------------------------------------------
# Plot the raw profiles together with the effective-medium estimate
# ---------------------------------------------------------------------------
# %%
fig, ax = plt.subplots(2, sharex=True, figsize=(6, 6))

ax[0].set_xlim(-10, 70)
ax[1].set_xlim(-10, 70)

ax[0].plot(z_par, eps_par, label="profile")
ax[0].axhline(eps_bulk, color="black", linestyle="dashed", label="bulk")
ax[0].set_ylabel(r"$\varepsilon_{\parallel}$")
draw_effective_box(ax[0], leff_par, eps_eff_par, label="effective-medium")
ax[0].legend(frameon=False)

ax[1].plot(z_perp, eps_perp_inv, label="profile")
ax[1].axhline(1 / eps_bulk, color="black", linestyle="dashed", label="bulk")
ax[1].set_xlabel(r"$z$ [$\AA$]")
ax[1].set_ylabel(r"$\varepsilon_{\perp}^{-1}$")
draw_effective_box(ax[1], leff_perp, 1 / eps_eff_perp_inv, label="effective-medium")
ax[1].legend(frameon=False)

fig.tight_layout()
plt.show()


# A short summary for the notebook or generated documentation.
print(f"Bulk estimate from the pore center: {eps_blk_par:.2f} ± {eps_blk_par_err:.2f}")
print(f"Effective length (parallel): {leff_par:.2f} ± {leff_par_err:.2f}")
print(f"Effective length (perpendicular): {leff_perp:.2f} ± {leff_perp_err:.2f}")


# %%
