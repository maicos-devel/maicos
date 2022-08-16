#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2022 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Tools for computing structure properties.

The structure modules of MAICoS allow for calculating Small-Angle X-ray
Scattering (SAXS) scattering intensities, scattering intensities using the
Debye equation, and dipolar order parameters from molecular simulation
trajectory files.
"""

import logging

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import capped_distance

from .. import tables
from ..decorators import make_whole, set_planar_class_doc, set_verbose_doc
from ..lib import sfactor
from ..utils import check_compound
from .base import AnalysisBase, PlanarBase


logger = logging.getLogger(__name__)


def compute_form_factor(q, atom_type):
    """Calculate the form factor for the given element for given q (1/Å).

    Handles united atom types like CH4 etc ...
    """
    element = tables.atomtypes[atom_type]

    if element == "CH1":
        form_factor = compute_form_factor(q, "C") + compute_form_factor(q, "H")
    elif element == "CH2":
        form_factor = compute_form_factor(
            q, "C") + 2 * compute_form_factor(q, "H")
    elif element == "CH3":
        form_factor = compute_form_factor(
            q, "C") + 3 * compute_form_factor(q, "H")
    elif element == "CH4":
        form_factor = compute_form_factor(
            q, "C") + 4 * compute_form_factor(q, "H")
    elif element == "NH1":
        form_factor = compute_form_factor(q, "N") + compute_form_factor(q, "H")
    elif element == "NH2":
        form_factor = compute_form_factor(
            q, "N") + 2 * compute_form_factor(q, "H")
    elif element == "NH3":
        form_factor = compute_form_factor(
            q, "N") + 3 * compute_form_factor(q, "H")
    else:
        form_factor = tables.CM_parameters[element].c
        q2 = (q / (4 * np.pi))**2
        for i in range(4):
            form_factor += tables.CM_parameters[element].a[i] * \
                np.exp(-tables.CM_parameters[element].b[i] * q2)

    return form_factor


@set_verbose_doc
class Saxs(AnalysisBase):
    """Compute SAXS scattering intensities.

    The q vectors are binned by their length using a binwidth given by -dq.
    Using the -nobin option the raw intensity for each q_{i,j,k} vector
    is saved using. Note that this only works reliable using constant
    box vectors! The possible scattering vectors q can be restricted by a
    miminal and maximal angle with the z-axis. For 0 and 180 all possible
    vectors are taken into account. For the scattering factor the structure
    factor is multiplied by a atom type specific form factor based on
    Cromer-Mann parameters. By using the -sel option atoms can be selected
    for which the profile is calculated. The selection uses the
    MDAnalysis selection commands.

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    noboindata : bool
        Do not bin the data. Only works reliable for NVT!
    startq : float
        Starting q (1/Å)
    endq : float
        Ending q (1/Å)
    dq : float
        binwidth (1/Å)
    mintheta float
        Minimal angle (°) between the q vectors and the z-axis.
    maxtheta : float
        Maximal angle (°) between the q vectors and the z-axis.
    output : str
        Output filename
    concfreq : int
        Default number of frames after which results are calculated
        and files refreshed. If `0` results are only calculated at
        the end of the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    results.q : np.ndarray
        length of binned q-vectors
    results.q_indices : np.ndarray
        Miller indices of q-vector (only if noboindata==True)
    results.scat_factor : np.ndarray
        Scattering intensities
    """

    def __init__(self,
                 atomgroup,
                 nobin=False,
                 startq=0,
                 endq=6,
                 dq=0.005,
                 mintheta=0,
                 maxtheta=180,
                 output="sq.dat",
                 concfreq=0,
                 **kwargs):
        super(Saxs, self).__init__(atomgroup, **kwargs)
        self.nobindata = nobin
        self.startq = startq
        self.endq = endq
        self.dq = dq
        self.mintheta = mintheta
        self.maxtheta = maxtheta
        self.output = output
        self.concfreq = concfreq

    def _prepare(self):

        self.mintheta = min(self.mintheta, self.maxtheta)
        self.maxtheta = max(self.mintheta, self.maxtheta)

        if self.mintheta < 0:
            logger.info(f"mintheta = {self.mintheta}° < 0°: "
                        "Set mininmal angle to 0°.")
            self.mintheta = 0
        if self.maxtheta > 180:
            logger.info(f"maxtheta = {self.maxtheta}° > 180°: "
                        "Set maximal angle to 180°.")
            self.maxtheta = np.pi

        self.mintheta *= np.pi / 180
        self.maxtheta *= np.pi / 180

        self.groups = []
        self.atom_types = []
        logger.info("\nMap the following atomtypes:")
        for atom_type in np.unique(self.atomgroup.types).astype(str):
            try:
                element = tables.atomtypes[atom_type]
            except KeyError:
                raise RuntimeError(f"No suitable element for '{atom_type}' "
                                   f"found. You can add '{atom_type}' "
                                   "together with a suitable element "
                                   "to 'share/atomtypes.dat'.")
            if element == "DUM":
                continue
            self.groups.append(
                self.atomgroup.select_atoms("type {}*".format(atom_type)))
            self.atom_types.append(atom_type)

            logger.info("{:>14} --> {:>5}".format(atom_type, element))

        if self.nobindata:
            self.box = np.diag(
                mda.lib.mdamath.triclinic_vectors(
                    self._universe.dimensions))
            self.q_factor = 2 * np.pi / self.box
            self.maxn = np.ceil(self.endq / self.q_factor).astype(int)
            self.S_array = np.zeros(list(self.maxn) + [len(self.groups)])
        else:
            self.n_bins = int(np.ceil((self.endq - self.startq) / self.dq))
            self.struct_factor = np.zeros([self.n_bins, len(self.groups)])

    def _single_frame(self):
        # Convert everything to cartesian coordinates.
        box = np.diag(mda.lib.mdamath.triclinic_vectors(self._ts.dimensions))
        for i, t in enumerate(self.groups):
            # map coordinates onto cubic cell
            positions = t.atoms.positions - box * \
                np.round(t.atoms.positions / box)
            q_ts, S_ts = sfactor.compute_structure_factor(
                np.double(positions), np.double(box), self.startq,
                self.endq, self.mintheta, self.maxtheta)

            S_ts *= compute_form_factor(q_ts, self.atom_types[i])**2

            if self.nobindata:
                self.S_array[:, :, :, i] += S_ts
            else:
                q_ts = q_ts.flatten()
                S_ts = S_ts.flatten()
                nonzeros = np.where(S_ts != 0)[0]

                q_ts = q_ts[nonzeros]
                S_ts = S_ts[nonzeros]

                struct_ts = np.histogram(q_ts,
                                         bins=self.n_bins,
                                         range=(self.startq, self.endq),
                                         weights=S_ts)[0]
                with np.errstate(divide='ignore', invalid='ignore'):
                    struct_ts /= np.histogram(q_ts,
                                              bins=self.n_bins,
                                              range=(self.startq, self.endq))[0]
                self.struct_factor[:, i] += np.nan_to_num(struct_ts)

    def _conclude(self):
        if self.nobindata:
            self.results.scat_factor = self.S_array.sum(axis=3)
            self.results.q_indices = np.array(
                list(np.ndindex(tuple(self.maxn))))
            self.results.q = np.linalg.norm(self.results.q_indices
                                            * self.q_factor[np.newaxis, :],
                                            axis=1)
        else:
            q = np.arange(self.startq, self.endq, self.dq) + 0.5 * self.dq
            nonzeros = np.where(self.struct_factor[:, 0] != 0)[0]
            scat_factor = self.struct_factor[nonzeros]

            self.results.q = q[nonzeros]
            self.results.scat_factor = scat_factor.sum(axis=1)

        self.results.scat_factor /= (self._index * self.atomgroup.n_atoms)

    def save(self):
        """Save the current profiles to a file."""
        if self.nobindata:
            out = np.hstack([
                self.results.q[:, np.newaxis], self.results.q_indices,
                self.results.scat_factor.flatten()[:, np.newaxis]])
            nonzeros = np.where(out[:, 4] != 0)[0]
            out = out[nonzeros]
            argsort = np.argsort(out[:, 0])
            out = out[argsort]

            boxinfo = "box_x = {0:.3f} Å, box_y = {1:.3f} Å, " \
                      "box_z = {2:.3f} Å\n".format(*self.box)
            self.savetxt(self.output, out,
                         columns=[boxinfo, "q (1/Å)", "q_i", "q_j",
                                  "q_k", "S(q) (arb. units)"])
        else:
            self.savetxt(self.output,
                         np.vstack([self.results.q,
                                    self.results.scat_factor]).T,
                         columns=["q (1/Å)", "S(q) (arb. units)"])


@set_verbose_doc
@set_planar_class_doc
@make_whole()
class Diporder(PlanarBase):
    """Calculate dipolar order parameters.

    Calculations include the projected dipole density
    P_0⋅ρ(z)⋅cos(θ[z]), the dipole orientation cos(θ[z]), the squared dipole
    orientation cos²(Θ[z]) and the number density ρ(z).

    Parameters
    ----------
    atomgroup : AtomGroup
       Atomgroup on which the analysis is executed
    ${PLANAR_CLASS_PARAMETERS}
    sym : bool
        symmetrize the profiles
    binmethod : str
        binning method: center of mass (COM) or center of charge (COC)
    ${MAKE_WHOLE_PARAMETER}
    output : str
        Output filename
    concfreq : int
        Default number of frames after which results are calculated
        and files refreshed. If `0` results are only calculated at the
        end of the analysis and not saved by default.
    ${VERBOSE_PARAMETER}

    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.P0 : np.ndarray
        P_0⋅ρ(z)⋅cos(θ[z]) [e/Å²]
    results.cos_theta : np.ndarray
        cos(θ[z])
    results.cos_2_theta : np.ndarray
        cos²(Θ[z])
    results.rho : np.ndarray
        ρ(z) [1/Å³]
    """

    def __init__(self,
                 atomgroup,
                 dim=2,
                 zmin=0,
                 zmax=None,
                 binwidth=1,
                 center=False,
                 comgroup=None,
                 sym=False,
                 binmethod='COM',
                 make_whole=True,
                 output="diporder.dat",
                 concfreq=0,
                 **kwargs):
        super(Diporder, self).__init__(atomgroups=atomgroup,
                                       dim=dim,
                                       zmin=zmin,
                                       zmax=zmax,
                                       binwidth=binwidth,
                                       center=center,
                                       comgroup=comgroup,
                                       **kwargs)
        self.sym = sym
        self.binmethod = binmethod
        self.make_whole = make_whole
        self.concfreq = concfreq
        self.output = output

    def _prepare(self):
        """Set things up before the analysis loop begins."""
        super(Diporder, self)._prepare()
        self.binmethod = self.binmethod.upper()
        if self.binmethod not in ["COM", "COC"]:
            raise ValueError('Unknown binning method: {}'.format(
                self.binmethod))

        # Check if all residues are identical.
        # Choose first residue as reference.
        residue_names = [ag.names for ag in self.atomgroup.split("residue")]
        for names in residue_names[1:]:
            if len(residue_names[0]) != len(names) and np.all(
                    residue_names[0] != names):
                raise ValueError("Not all residues are identical. Please adjust"
                                 "selection.")

        # Assume a threedimensional universe...
        self.xydims = np.roll(np.arange(3), -self.dim)[1:]
        self.P0 = np.zeros(self.n_bins)
        self.cos_theta = np.zeros(self.n_bins)
        self.cos_2_theta = np.zeros(self.n_bins)
        self.rho = np.zeros(self.n_bins)
        self.bin_count = np.zeros(self.n_bins)

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

    def _single_frame(self):
        super(Diporder, self)._single_frame()
        dz_frame = self._ts.dimensions[self.dim] / self.n_bins

        chargepos = self.atomgroup.positions \
            * self.atomgroup.charges[:, np.newaxis]
        dipoles = self.atomgroup.accumulate(
            chargepos, compound=check_compound(self.atomgroup))

        if self.binmethod == 'COM':
            # Calculate the centers of the objects (i.e. Molecules)
            bin_positions = self.atomgroup.center_of_mass(
                compound=check_compound(self.atomgroup))
        elif self.binmethod == 'COC':
            bin_positions = self.atomgroup.center(
                weights=np.abs(self.atomgroup.charges),
                compound=check_compound(self.atomgroup))

        bincount = np.histogram(bin_positions[:, self.dim],
                                bins=self.n_bins,
                                range=(self.zmin, self.zmax))[0]
        self.bin_count += bincount
        A = np.prod(self._ts.dimensions[self.xydims])

        self.P0 += np.histogram(
            bin_positions[:, self.dim],
            bins=self.n_bins,
            range=(self.zmin, self.zmax),
            weights=np.dot(dipoles, self.unit))[0] / (A * dz_frame)
        self.cos_theta += np.histogram(
            bin_positions[:, self.dim],
            bins=self.n_bins,
            range=(self.zmin, self.zmax),
            weights=np.dot(
                dipoles / np.linalg.norm(dipoles, axis=1)[:, np.newaxis],
                self.unit))[0]
        self.cos_2_theta += np.histogram(
            bin_positions[:, self.dim],
            bins=self.n_bins,
            range=(self.zmin, self.zmax),
            weights=np.dot(
                dipoles / np.linalg.norm(dipoles, axis=1)[:, np.newaxis],
                self.unit)**2)[0]
        self.rho += bincount / (A * dz_frame)

    def _conclude(self):
        """Calculate the results.

        Called at the end of the run() method to before the _conclude function.
        Can also be called during a run to update the results during
        processing.
        """
        super(Diporder, self)._conclude()
        self.results.P0 = self.P0 / self._index
        self.results.rho = self.rho / self._index

        with np.errstate(divide='ignore', invalid='ignore'):
            self.results.cos_theta = np.nan_to_num(
                self.cos_theta / self.bin_count)
            self.results.cos_2_theta = np.nan_to_num(
                self.cos_2_theta / self.bin_count)

        if self.sym:
            for i in range(len(self.results.z) - 1):
                self.results.z[i + 1] \
                    = .5 * (self.results.z[i + 1]
                            + self.results.z[i + 1][-1::-1])
                self.results.diporder[i + 1] \
                    = .5 * (self.results.diporder[i + 1]
                            + self.results.diporder[i + 1][-1::-1])

    def save(self):
        """Save results to a file."""
        header = ["z [Å]"]
        header.append("P_0*rho(z)*cos(Theta[z]) [e/Å^2]")
        header.append("cos(theta(z))")
        header.append("cos^2(theta(z))")
        header.append("rho(z) [1/Å^3]")

        self.savetxt(self.output,
                     np.vstack([
                         self.results.z, self.results.P0,
                         self.results.cos_theta, self.results.cos_2_theta,
                         self.results.rho
                         ]).T,
                     columns=header)


@set_verbose_doc
@set_planar_class_doc
class RDFPlanar(PlanarBase):
    r"""Compute slab wise planar two dimensional radial distribution functions.

    The radial distribution function :math:`g_\text{planar}(r)` describes the
    spatial correlation between atoms in `g1` and atoms in `g2`. The 2D RDF can
    be used in systems that are inhomogeneous along one axis, and homogeneous
    along a plane. It gives the average number density of `g2` as a
    function of lateral distance from a centered `g1` atom. In fully
    homogeneous systems and in the limit of small 'dzheight' :math:`\Delta z`,
    it is the same as the well known three dimensional RDF.

    .. math::

     g_\text{planar}(r) =
     \frac{1}{N_{g1}2 \Delta z} \cdot \sum_{i}^{N_{g1}} \sum_{j}^{N_{g2}}
     \delta(r - r_{ij}) \cdot \left( \theta \left(|z_{ij}| + {\Delta z}
     \right) - \theta \left( |z_{ij}| - {\Delta z} \right) \right) .

    As the density to normalise the RDF with is unknown, the output
    is in the dimension of number/volume in 1/Å^3.

    Functionally, RDFPlanar bins all pairwise `g1`-`g2` distances
    where the z distance is smaller than `dzheight` in a histogram.

    Parameters
    ----------
    g1 : AtomGroup
        First AtomGroup
    g2 : AtomGroup
        Second AtomGroup
    rdf_binwidth : int
        Binwidth of bins in the histogram of the RDF (Å)
    dzheight : float
        dz height of a RDF slab (Å)
    range: (float, float)
        the minimum and maximum pairwise distance between 'g1' and 'g2'. (Å)
    binmethod : str
        Method for position binning; possible options are
        center of geometry (cog), center of mass (com) or
        center of charge (coc).
    ${PLANAR_CLASS_PARAMETERS}
    ${VERBOSE_PARAMETER}


    Attributes
    ----------
    ${PLANAR_CLASS_ATTRIBUTES}
    results.bins: np.ndarray
        distances to which the RDF is calculated with shape (rdf_nbins) (Å)
    results.rdf: np.ndrray
        RDF with shape (rdf_nbins, n_bins) (1/Å^3)
    """

    def __init__(self,
                 g1,
                 g2=None,
                 rdf_binwidth=0.3,
                 dzheight=0.1,
                 range=(0.0, None),
                 binmethod="com",
                 # Planar base arguments
                 dim=2,
                 binwidth=1,
                 center=False,
                 comgroup=None,
                 zmin=0,
                 zmax=None,
                 output="rdf.dat",
                 concfreq=0,
                 **kwargs):

        super(RDFPlanar, self).__init__(atomgroups=g1,
                                        dim=dim,
                                        zmin=zmin,
                                        zmax=zmax,
                                        binwidth=binwidth,
                                        center=center,
                                        comgroup=comgroup,
                                        concfreq=concfreq,
                                        **kwargs)

        self.g1 = g1
        if g2 is None:
            self.g2 = g1
        else:
            self.g2 = g2
        self.range = range
        self.rdf_binwidth = rdf_binwidth
        self.dzheight = dzheight
        self.output = output
        self.binmethod = binmethod.lower()

    def _prepare(self):
        super(RDFPlanar, self)._prepare()
        if self._verbose:
            logger.info('Compute radial distribution function.')

        self.odims = np.roll(np.arange(3), -self.dim)[1:]
        half_of_box_size = min(self._universe.dimensions[self.odims]) / 2
        if self.range[1] is None:
            self.range = (self.range[0], half_of_box_size)
            logger.info("Setting maximum range of RDF to half the box size"
                        f"({self.range[1]} Å).")
        elif self.range[1] > half_of_box_size:
            raise ValueError("Range of RDF exceeds half of the box size. "
                             f"Set to smaller than {half_of_box_size} Å.")

        try:
            if self.rdf_binwidth > 0:
                self.rdf_nbins = int(np.ceil((self.range[1] - self.range[0])
                                     / self.rdf_binwidth))
            else:
                raise ValueError("RDF binwidth must be a positive number.")
        except TypeError:
            raise ValueError("RDF binwidth must be a number.")

        if self.binmethod not in ["cog", "com", "coc"]:
            raise ValueError(f"{self.binmethod} is an unknown binning "
                             "method. Use `cog`, `com` or `coc`.")

        logger.info(f"Using {self.rdf_nbins} rdf bins.")

        # Empty histogram self.count to store the RDF.
        self.count = np.zeros((self.n_bins, self.rdf_nbins))
        self.edges = np.histogram([-1], bins=self.rdf_nbins,
                                  range=self.range)[1]
        self.bins = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.n_g1_total = np.zeros((self.n_bins, 1))

        # Set the max range to filter the search radius.
        self._maxrange = self.range[1]

    def _single_frame(self):
        super(RDFPlanar, self)._single_frame()
        binwidth = (self.zmax - self.zmin) / self.n_bins

        if self.binmethod == 'com':
            g1_bin_positions = self.g1.center_of_mass(
                compound=check_compound(self.g1))
            g2_bin_positions = self.g2.center_of_mass(
                compound=check_compound(self.g2))
        elif self.binmethod == 'coc':
            g1_bin_positions = self.g1.center_of_charge(
                compound=check_compound(self.g1))
            g2_bin_positions = self.g2.center_of_charge(
                compound=check_compound(self.g2))
        elif self.binmethod == 'cog':
            g1_bin_positions = self.g1.center_of_geometry(
                compound=check_compound(self.g1))
            g2_bin_positions = self.g2.center_of_geometry(
                compound=check_compound(self.g2))

        # Calculate planar rdf per bin by averaging over all atoms in one bin.
        for z_bin in range(0, self.n_bins):
            # Set zmin and zmax of the bin.
            z_min = self.zmin + binwidth * z_bin
            z_max = self.zmin + binwidth * (z_bin + 1)

            # Get all atoms in a bin.
            g1_in_zbin_positions = g1_bin_positions[np.logical_and(
                g1_bin_positions[:, self.dim] >= z_min,
                g1_bin_positions[:, self.dim] < z_max)]

            g2_in_zbin_positions = g2_bin_positions[np.logical_and(
                g2_bin_positions[:, self.dim] >= z_min - self.dzheight,
                g2_bin_positions[:, self.dim] < z_max + self.dzheight)]

            n_g1 = len(g1_in_zbin_positions)
            n_g2 = len(g2_in_zbin_positions)
            self.n_g1_total[z_bin] += n_g1

            # Extract z coordinate.
            z_g1 = np.copy(g1_in_zbin_positions)
            z_g2 = np.copy(g2_in_zbin_positions)
            # Set other coordinates to 0.
            z_g1[:, self.odims] = 0
            z_g2[:, self.odims] = 0

            # Automatically filter only those pairs with delta z < dz.
            z_pairs, _ = capped_distance(z_g1,
                                         z_g2,
                                         self.dzheight,
                                         box=self._universe.dimensions)

            # Calculate pairwise distances between g1 and g2.
            pairs, xy_distances = capped_distance(g1_in_zbin_positions,
                                                  g2_in_zbin_positions,
                                                  self._maxrange,
                                                  box=self.
                                                  _universe.dimensions)

            # Map pairs (i, j) to a number i+N*j (so we can use np.isin).
            z_pairs_encode = z_pairs[:, 0] + n_g2 * z_pairs[:, 1]
            pairs_encode = pairs[:, 0] + n_g2 * pairs[:, 1]

            mask_in_dz = np.isin(pairs_encode, z_pairs_encode)
            mask_different_atoms = np.where(xy_distances > 0, True, False)

            relevant_xy_distances = xy_distances[mask_in_dz
                                                 * mask_different_atoms]
            # Histogram the pairwise distances.
            self.count[z_bin] += np.histogram(relevant_xy_distances,
                                              bins=self.rdf_nbins,
                                              range=self.range)[0]

    def _conclude(self):
        super(RDFPlanar, self)._conclude()

        # Normalise rdf using the volumes of a ring with height dz.
        ring_volumes = (np.pi * (self.edges[1:]**2 - self.edges[:-1]**2)
                        * self.dzheight)
        ring_volumes = np.expand_dims(ring_volumes, axis=0)
        self.results.bins = self.bins
        self.results.rdf = self.count / self.n_g1_total / ring_volumes / 2
        self.results.rdf = np.nan_to_num(self.results.rdf.T, nan=0)

    def save(self):
        """Save results."""
        columns = ["r [Å]"]
        for z in self.results.z:
            columns.append(f"rdf at {z:.2f} Å [Å^-3]")

        self.savetxt(self.output,
                     np.hstack([self.results.bins[:, np.newaxis],
                                self.results.rdf]),
                     columns=columns)
