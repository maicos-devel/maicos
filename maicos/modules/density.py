#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later
r"""

The density modules of MAICoS are tools for computing density, 
temperature, and chemical potential profiles from molecular 
simulation trajectory files. Profiles can be extracted either 
in Cartesian or cylindrical coordinate systems. Units for the density 
are the same as GROMACS, i.e. mass, number, charge, and electron. 
See the `gmx density`_ manual for details.

**From the command line**

You can extract a density profile from your molecular dynamics 
trajectories directly from the terminal. For this example, we use 
the ``airwater`` data file of MAICoS. First, go to the directory

.. code-block:: bash

    cd tests/data/airwater/

then type:

.. code-block:: bash

    maicos density_planar -s conf.gro -traj traj.trr

Here ``conf.gro`` and ``traj.trr`` are GROMACS configuration and 
trajectory files, respectively. The density profile appears in 
a ``.dat`` file. You can visualise all the options of the module 
``density_planar`` by typing

.. code-block:: bash

    maicos density_planar -h

**From the Python interpreter**

In order to calculate the density using MAICoS in a Python environment, 
first import MAICoS and MDAnalysis:

.. code-block:: python3

	import MDAnalysis as mda
	import maicos
	
Then create a MDAnalysis universe:

.. code-block:: python3

	u = mda.Universe('conf.gro', 'traj.trr')
	grpH2O = u.select_atoms('type O or type H')

And run MAICoS' density_planar module:
	
.. code-block:: python3

	dplan = maicos.density_planar(grpH2O)
	dplan.run()   	

Results can be accessed from ``dplan.results``. More details are 
given in the :ref:`tutorial <label_tutorial_density_planar>` below. 

.. _`gmx density`: https://manual.gromacs.org/archive/5.0.7/programs/gmx-density.html

|
"""

import warnings

import numpy as np
from scipy import constants

from .base import MultiGroupAnalysisBase
from ..utils import savetxt, atomgroup_header


def mu(rho, temperature, m):
    """Returns the chemical potential calculated from the density: mu = k_B T log(rho. / m)"""

    # De Broglie (converted to nm)
    db = np.sqrt(constants.h**2 / (2 * np.pi * m * constants.atomic_mass *
                                   constants.Boltzmann * temperature))

    # kT in KJ/mol
    kT = temperature * constants.Boltzmann * constants.Avogadro / constants.kilo

    if np.all(rho > 0):
        return kT * np.log(rho * db**3 / (m * constants.atomic_mass))
    elif np.any(rho == 0):
        return np.float64("-inf")
    else:
        return np.float("nan")


def dmu(rho, drho, temperature):
    """Returns the error of the chemical potential calculated from the density using propagation of uncertainty."""

    if np.all(rho > 0):
        return (drho / rho)
    elif np.any(rho == 0):
        return np.float64("-inf")
    else:
        return np.float("nan")


def weight(selection, dens):
    """Calculates the weights for the histogram depending on the choosen type of density.
        Valid values are `mass`, `number`, `charge` or `temp`."""
    if dens == "mass":
        # amu/nm**3 -> kg/m**3
        return selection.atoms.masses * constants.atomic_mass * 1e27
    elif dens == "number":
        return np.ones(selection.atoms.n_atoms)
    elif dens == "charge":
        return selection.atoms.charges
    elif dens == "temp":
        # ((1 amu * Angstrom^2) / (picoseconds^2)) / Boltzmann constant
        prefac = constants.atomic_mass * 1e4 / constants.Boltzmann
        return ((selection.atoms.velocities**2).sum(axis=1) *
                selection.atoms.masses / 2 * prefac)
    else:
        raise ValueError(
            "`{}` not supported. Use `mass`, `number`, `charge` or `temp`".
            format(dens))


class density_planar(MultiGroupAnalysisBase):
    """ The density planar module computes partial densities or 
temperature profiles in the Cartesian coordinate system.
    
    **Tutorial**

    .. _label_tutorial_density_planar:

    This tutorial has been written for the version ``0.3`` of MAICoS. 
To follow this tutorial, the data test files of MAICoS are needed. 
From a terminal, download MAICoS at a location of your choice:

    .. code-block:: bash

	    cd mypath
	    git clone git@gitlab.com:maicos-devel/maicos.git

    In a python environment, import MDAnalysis, MAICoS, PyPlot, and NumPy:

    .. code-block:: python3

	    import MDAnalysis as mda
	    import maicos
	    import matplotlib.pyplot as plt
	    import numpy as np
	    

    Define the path to the ``airwater`` data folder of MAICoS:

    .. code-block:: python3

	    datapath = 'mypath/maicos/tests/data/airwater/'
	    
    The system consists of a 2D slab with 352 water molecules in vacuum, 
where the two water/vacuum interfaces are normal to the axis :math:`z`:


    .. image:: ../images/airwater.png
       :width: 600

    Create a universe using MDAnalysis and define a group containing 
the oxygen and the hydrogen atoms of the water molecules:

    .. code-block:: python3

	    u = mda.Universe(datapath+'conf.gro', datapath+'traj.trr')
	    grpH2O = u.select_atoms('type O or type H')

    Let us call the ``density_planar`` module:

    .. code-block:: python3

	    dplan = maicos.density_planar(grpH2O)
	    dplan.run()   

    Extract the coordinate and the density profile:

    .. code-block:: python3

	    zcoor = dplan.results['z']
	    dens = dplan.results['dens_mean']

    By default the binwidth is 0.1 nanometers, the units are kg/m3, 
and the axis is :math:`z`. Plot it using 

    .. code-block:: python3

	    fig = plt.figure(figsize = (12,6))
	    plt.plot(zcoor,dens,linewidth=2)
	    plt.xlabel("z coordinate [nanometer]")
	    plt.ylabel("density H2O [kg/m3]")
	    plt.show()
	    
    .. image:: ../images/density_planar.png
       :width: 600

    
    They are several options you can play with. To know the full 
list of options, have a look at the ``Inputs`` section below. 
For instance, you can increase the spacial resolution 
by reducing the binwidth:

    .. code-block:: python3 

	    dplan = maicos.density_planar(grp_oxy, binwidth = 0.05) 

    **Inputs**

    :param output (str): Output filename
    :param outfreq (int): Default time after which output files are refreshed (1000 ps).
    :param dim (int): Dimension for binning (0=X, 1=Y, 2=Z)
    :param binwidth (float): binwidth (nanometer)
    :param mu (bool): Calculate the chemical potential
    :param muout (str): Prefix for output filename for chemical potential
    :param temperature (float): temperature (K) for chemical potential
    :param mass (float): atommass if not guessed from topology
    :param zpos (float): position at which the chemical potential will be computed. By default average over box.
    :param dens (str): Density: mass, number, charge, electron
    :param comgroup (str): Perform the binning relative to the center of mass of the selected group.
    :param center (bool): Perform the binning relative to the center of the (changing) box.

    **Outputs**

    :returns (dict): * z: bins
                    * deans_mean: calculated densities
                    * dens_err: density error
                    * mu: chemical potential
                    * dmu: error of chemical potential
       
    |
    """

    def __init__(self,
                 atomgroups,
                 output="density.dat",
                 outfreq=1000,
                 dim=2,
                 binwidth=0.1,
                 mu=False,
                 muout="muout.dat",
                 temperature=300,
                 mass=np.nan,
                 zpos=None,
                 dens="mass",
                 comgroup=None,
                 center=False,
                 **kwargs):
        super().__init__(atomgroups, **kwargs)
        self.output = output
        self.outfreq = outfreq
        self.dim = dim
        self.binwidth = binwidth
        self.mu = mu
        self.muout = muout
        self.temperature = temperature
        self.mass = mass
        self.zpos = zpos
        self.dens = dens
        self.comgroup = comgroup
        self.center = center

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-d', dest='dim')
        parser.add_argument('-dz', dest='binwidth')
        parser.add_argument('-mu', dest='mu')
        parser.add_argument('-muo', dest='muout')
        parser.add_argument('-temp', dest='temperature')
        parser.add_argument('-zpos', dest='zpos')
        parser.add_argument('-dens', dest='dens')
        parser.add_argument('-com', dest='comgroup')
        parser.add_argument('-center', dest='center')

    def _prepare(self):
        if self.dens not in ["mass", "number", "charge", "temp"]:
            raise ValueError(
                "Invalid choice for dens: '{}' (choose from 'mass', "
                "'number', 'charge', 'temp')".format(self.dens))
        if self._verbose:
            if self.dens == 'temp':
                print('Computing temperature profile along {}-axes.'.format(
                    'XYZ'[self.dim]))
            else:
                print('Computing {} density profile along {}-axes.'.format(
                    self.dens, 'XYZ'[self.dim]))

        self.ngroups = len(self.atomgroups)
        self.nbins = int(
            np.ceil(self._universe.dimensions[self.dim] / 10 / self.binwidth))

        self.density_mean = np.zeros((self.nbins, self.ngroups))
        self.density_mean_sq = np.zeros((self.nbins, self.ngroups))
        self.av_box_length = 0

        if self.mu and self.dens != 'mass':
            raise ValueError(
                "Calculation of the chemical potential is only possible when "
                "mass density is selected")

        if self.mu:
            if len(self.atomgroups) != 1:
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    warnings.warn(
                        "Performing chemical potential analysis for 1st selection"
                        "group '{}'".format(self.atomgroups[0]))
            self.mass = self.atomgroups[0].total_mass(
            ) / self.atomgroups[0].atoms.n_residues

        if self.comgroup is not None:
            self.comsel = self._universe.select_atoms(self.comgroup)
            if self._verbose:
                print("{:>15}: {:>10} atoms".format(self.comgroup,
                                                    self.comsel.n_atoms))
            if self.comsel.n_atoms == 0:
                raise ValueError(
                    "`{}` does not contain any atoms. Please adjust 'com' selection."
                    .format(self.comgroup))
        if self.comgroup is not None:
            self.center = True  # always center when COM
        if self._verbose:
            print("\n")
            print('Using', self.nbins, 'bins.')

    def _single_frame(self):
        curV = self._ts.volume / 1000
        self.av_box_length += self._ts.dimensions[self.dim] / 10
        """ center of mass calculation with generalization to periodic systems
        see Bai, Linge; Breen, David (2008). "Calculating Center of Mass in an
        Unbounded 2D Environment". Journal of Graphics, GPU, and Game Tools. 13
        (4): 53–60. doi:10.1080/2151237X.2008.10129266,
        https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
        """
        if self.comgroup is None:
            comshift = 0
        else:
            Theta = self.comsel.positions[:, self.dim] / self._ts.dimensions[
                self.dim] * 2 * np.pi
            Xi = (np.cos(Theta) *
                  self.comsel.masses).sum() / self.comsel.masses.sum()
            Zeta = (np.sin(Theta) *
                    self.comsel.masses).sum() / self.comsel.masses.sum()
            ThetaCOM = np.arctan2(-Zeta, -Xi) + np.pi
            comshift = self._ts.dimensions[self.dim] * (0.5 - ThetaCOM /
                                                        (2 * np.pi))

        dz = self._ts.dimensions[self.dim] / self.nbins

        for index, selection in enumerate(self.atomgroups):
            bins = (np.rint(
                (selection.atoms.positions[:, self.dim] + comshift + dz / 2) /
                dz) % self.nbins).astype(int)
            density_ts = np.histogram(bins,
                                      bins=np.arange(self.nbins + 1),
                                      weights=weight(selection, self.dens))[0]

            if self.dens == 'temp':
                bincount = np.bincount(bins, minlength=self.nbins)
                self.density_mean[:, index] += density_ts / bincount
                self.density_mean_sq[:, index] += (density_ts / bincount)**2
            else:
                self.density_mean[:, index] += density_ts / curV * self.nbins
                self.density_mean_sq[:, index] += (density_ts / curV *
                                                   self.nbins)**2

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["dens_mean"] = self.density_mean / self._index
        self.results["dens_mean_sq"] = self.density_mean_sq / self._index

        self.results["dens_std"] = np.nan_to_num(
            np.sqrt(self.results["dens_mean_sq"] -
                    self.results["dens_mean"]**2))
        self.results["dens_err"] = self.results["dens_std"] / \
            np.sqrt(self._index)

        dz = self.av_box_length / (self._index * self.nbins)
        if self.center:
            self.results["z"] = np.linspace(
                -self.av_box_length / self._index / 2,
                self.av_box_length / self._index / 2,
                self.nbins,
                endpoint=False) + dz / 2
        else:
            self.results["z"] = np.linspace(
                0, self.av_box_length / self._index, self.nbins,
                endpoint=False) + dz / 2

        # chemical potential
        if self.mu:
            if (self.zpos is not None):
                this = (np.rint(
                    (self.zpos + dz / 2) / dz) % self.nbins).astype(int)
                if self.center:
                    this += np.rint(self.nbins / 2).astype(int)
                self.results["mu"] = mu(self.results["dens_mean"][this][0],
                                        self.temperature, self.mass)
                self.results["dmu"] = dmu(self.results["dens_mean"][this][0],
                                          self.results["dens_err"][this][0],
                                          self.temperature)
            else:
                self.results["mu"] = np.mean(
                    mu(self.results["dens_mean"], self.temperature, self.mass))
                self.results["dmu"] = np.mean(
                    dmu(self.results["dens_mean"], self.results["dens_err"],
                        self.temperature))

    def _save_results(self):
        # write header
        if self.dens == "mass":
            units = "kg m^(-3)"
        elif self.dens == "number":
            units = "nm^(-3)"
        elif self.dens == "charge":
            units = "e nm^(-3)"
        elif self.dens == "temp":
            units = "K"

        if self.dens == 'temp':
            columns = "temperature profile [{}]".format(units)
        else:
            columns = "{} density profile [{}]".format(self.dens, units)
        columns += "\nstatistics over {:.1f} picoseconds \npositions [nm]".format(
            self._index * self._universe.trajectory.dt)
        try:
            for group in self.atomgroups:
                columns += "\t" + atomgroup_header(group)
            for group in self.atomgroups:
                columns += "\t" + atomgroup_header(group) + " error"
        except AttributeError:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn("AtomGroup does not contain resnames."
                              " Not writing residues information to output.")

        # save density profile
        savetxt(self.output,
                np.hstack(
                    ((self.results["z"][:, np.newaxis]),
                     self.results["dens_mean"], self.results["dens_err"])),
                header=columns)

        if self.mu:
            # save chemical potential
            savetxt(self.muout,
                    np.hstack((self.results["mu"], self.results["dmu"]))[None],
                    header="μ [kJ/mol]\t μ error [kJ/mol]")


class density_cylinder(MultiGroupAnalysisBase):
    """Computes partial densities across a cylinder of given radius r and length l

    **Inputs**
     
    :param output (str): Output filename
    :param outfreq (int): Default time after which output files are refreshed (1000 ps).
    :param dim (int): Dimension for binning (0=X, 1=Y, 2=Z)
    :param center (str): Perform the binning relative to the center of this selection string of teh first AtomGroup. If None center of box is used.
    :param radius (float): Radius of the cylinder (nm). If None smallest box extension is taken.
    :param binwidth (float): binwidth (nanometer)
    :param length (float): Length of the cylinder (nm). If None length of box in the binning dimension is taken.
    :param dens (str): Density: mass, number, charge, temp

    **Outputs**

    :returns (dict): * z: bins
                     * deans_mean: calculated densities
                     * dens_err: density error
        """

    def __init__(self,
                 atomgroups,
                 output="density_cylinder.dat",
                 outfreq=1000,
                 dim=2,
                 center=None,
                 radius=None,
                 binwidth=0.1,
                 length=None,
                 dens="mass",
                 **kwargs):
        super().__init__(atomgroups, **kwargs)
        self.output = output
        self.outfreq = outfreq
        self.dim = dim
        self.binwidth = binwidth
        self.center = center
        self.radius = radius
        self.length = length
        self.dens = dens

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-d', dest='dim')
        parser.add_argument('-center', dest='center')
        parser.add_argument('-r', dest='radius')
        parser.add_argument('-dr', dest='binwidth')
        parser.add_argument('-l', dest='length')
        parser.add_argument('-dens', dest='dens')

    def _prepare(self):

        if self.dens not in ["mass", "number", "charge", "temp"]:
            raise ValueError(
                "Invalid choice for dens: '{}' (choose from 'mass', "
                "'number', 'charge', 'temp')".format(self.dens))

        if self._verbose:
            if self.dens == 'temp':
                print('Computing temperature profile along {}-axes.'.format(
                    'XYZ'[self.dim]))
            else:
                print(
                    'Computing radial {} density profile along {}-axes.'.format(
                        self.dens, 'XYZ'[self.dim]))

        self.odims = np.roll(np.arange(3), -self.dim)[1:]

        if self.center is None:
            if self._verbose:
                print("No center given --> Take from box dimensions.")
            self.centersel = None
            center = self.atomgroups[0].dimensions[:3] / 2
        else:
            self.centersel = self.atomgroups[0].select_atoms(self.center)
            if len(self.centersel) == 0:
                raise RuntimeError("No atoms found in center selection. "
                                   "Please adjust selection!")
            center = self.centersel.center_of_mass()

        if self._verbose:
            print("Initial center at {}={:.3f} nm and {}={:.3f} nm.".format(
                'XYZ'[self.odims[0]], center[self.odims[0]] / 10,
                'XYZ'[self.odims[1]], center[self.odims[1]] / 10))

        if self.radius is None:
            self.radius = self.atomgroups[0].dimensions[self.odims].min() / 2
            if self._verbose:
                print(
                    "No radius given --> Take smallest box extension (r={:.2f} nm)."
                    .format(self.radius / 10))
        else:
            self.radius /= 10

        if self.length is None:
            self.length = self.atomgroups[0].dimensions[self.dim]
            if self._verbose:
                print("No length given --> Take length in {}.".format(
                    'XYZ'[self.dim]))
        else:
            self.length /= 10

        self.ngroups = len(self.atomgroups)
        self.nbins = int(np.ceil(self.radius / 10 / self.binwidth))

        self.density_mean = np.zeros((self.nbins, self.ngroups))
        self.density_mean_sq = np.zeros((self.nbins, self.ngroups))

        self._dr = np.ones(self.nbins) * self.radius / self.nbins
        self._r_bins = np.arange(self.nbins) * self._dr + self._dr
        self._delta_r_sq = self._r_bins**2 - \
            np.insert(self._r_bins, 0, 0)[0:-1]**2  # r_o^2 - r_i^2

        if self._verbose:
            print("\n")
            print('Using', self.nbins, 'bins.')

    def _single_frame(self):
        # calculater center of cylinder.
        if self.center is None:
            center = self.atomgroups[0].dimensions[:3] / 2
        else:
            center = self.centersel.center_of_mass()

        for index, selection in enumerate(self.atomgroups):

            # select cylinder of the given length and radius
            cut = selection.atoms[np.where(
                np.absolute(selection.atoms.positions[:, self.dim] -
                            center[self.dim]) < self.length / 2)[0]]
            cylinder = cut.atoms[np.where(
                np.linalg.norm((cut.atoms.positions[:, self.odims] -
                                center[self.odims]),
                               axis=1) < self.radius)[0]]

            radial_positions = np.linalg.norm(
                (cylinder.atoms.positions[:, self.odims] - center[self.odims]),
                axis=1)
            bins = np.digitize(radial_positions, self._r_bins)
            density_ts = np.histogram(bins,
                                      bins=np.arange(self.nbins + 1),
                                      weights=weight(cylinder, self.dens))[0]

            if self.dens == 'temp':
                bincount = np.bincount(bins, minlength=self.nbins)
                self.density_mean[:, index] += density_ts / bincount
                self.density_mean_sq[:, index] += (density_ts / bincount)**2
            else:
                self.density_mean[:, index] += density_ts * 1000 / (
                    np.pi * self._delta_r_sq * self.length)
                self.density_mean_sq[:, index] += (
                    density_ts * 1000 /
                    (np.pi * self._delta_r_sq * self.length))**2

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results["r"] = (np.copy(self._r_bins) - self._dr / 2) / 10
        self.results["dens_mean"] = self.density_mean / self._index
        self.results["dens_mean_sq"] = self.density_mean_sq / self._index

        self.results["dens_std"] = np.nan_to_num(
            np.sqrt(self.results["dens_mean_sq"] -
                    self.results["dens_mean"]**2))
        self.results["dens_err"] = self.results["dens_std"] / np.sqrt(
            self._index)

    def _save_results(self):
        # write header
        if self.dens == "mass":
            units = "kg m^(-3)"
        elif self.dens == "number":
            units = "nm^(-3)"
        elif self.dens == "charge":
            units = "e nm^(-3)"
        elif self.dens == "temp":
            units = "K"

        if self.dens == 'temp':
            columns = "temperature profile [{}]".format(units)
        else:
            columns = "{} density profile [{}]".format(self.dens, units)
        columns += "\nstatistics over {:.1f} picoseconds \npositions [nm]".format(
            self._index * self._universe.trajectory.dt)
        for group in self.atomgroups:
            columns += "\t" + atomgroup_header(group)
        for group in self.atomgroups:
            columns += "\t" + atomgroup_header(group) + " error"

        # save density profile
        savetxt(self.output,
                np.hstack(
                    ((self.results["r"][:, np.newaxis]),
                     self.results["dens_mean"], self.results["dens_err"])),
                header=columns)
