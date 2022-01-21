#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2019 Authors and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
# SPDX-License-Identifier: GPL-2.0-or-later

import numpy as np

from ..utils import check_compound, savetxt
from .base import SingleGroupAnalysisBase


class dipole_angle(SingleGroupAnalysisBase):
    """Calculate angle timeseries of dipole moments with respect to an axis.
    
    The dipole angle function in the timeseries module computes the dipole moment of an MD simulation trajectory with respect to a reference axis. 

    The program can be run from either a python environment or a command line interface.
    As an example, we analyse a box of water molecules simulated for 100 picoseconds (ps) using an NVT ensemble. 
    The length of the box was 5.45 nm containing around 5000 water molecules. 
    The time step of simulation was 2 fs. 
    Periodic boundary conditions were employed in all directions and long range electrostatics were modelled using the PME method. 
    LINCS algorithm was used to constraint the H-Bonds at a temperature of 300K. 
    A pulsed and alternating electric field was applied along the x-axis in the form of a guassian laser pulse. 
    Please check `gromacs electric field`_ for more details. 

    The simulation directory can be downloaded from the repository (shown below) which contains ``mdelectric.tpr`` and ``mdelectric.trr``

    .. code-block:: bash

        cd tests/data/electricfwater

    **From the python interpreter**

    We need to import `MD Analysis`_, `matplotlib`_ and maicos packages in the environment

    .. code-block:: python3

        import MDAnalysis as mda
        import matplotlib.pyplot as plt
        import maicos

    Then create an MD Analysis Universe. 

    .. code-block:: python3

        u = mda.Universe("mdelectric.tpr","mdelectric.trr")
        at = u.atoms

    Now run the MAICOS dipole angle function.

    .. code-block:: python3

        dipangle = maicos.dipole_angle(at, dim=0)
        dipangle.run()

    The option ``dim = 0`` specifies the reference vector  ``x-axis``.
    The run produces a `python dictionary` named ``dipangle.results`` with `4 keys` linked to `numpy arrays` as `values`. 
    They are timestep, cosine of dipole and x-axis, cosine squared, product of cosine of dipoles i and j (i!=j)

    The results can be visualized as follows:

    .. code-block:: python3

        plt.plot(dipangle.results["t"],dipangle.results["cos_theta_i"])
        plt.title("Average cos between dipole and x-axis")
        plt.xlabel("Time (ps)")
        plt.ylabel(r'cos($\theta_i$)')
        plt.show()

    The figure generated :

       .. image:: ../images/dipangle.png
        :width: 600

    MAICOS can also be accessed through command line interface.

    **From the command line interface**

    .. code-block:: bash

        maicos dipole_angle -s md.tpr -f md.trr -d 0

    The output file ``dipangle.dat`` is similar to `dipangle.results` and contains the data in columns. 

    They are several options you can play with. To know the full 
    list of options, have a look at the ``Inputs`` section below. 

    .. _`MD Analysis`: https://www.mdanalysis.org/
    .. _`matplotlib`: https://matplotlib.org/ 
    .. _`gromacs electric field`: https://manual.gromacs.org/2019-current/reference-manual/special/electric-fields.html#fig-field


    **Inputs**

    :param dim (int): refernce vector for angle (x=0, y=1, z=2)
    :param outfreq (float): Default number of frames after which output files are refreshed
    :param output (str): Prefix for output filenames

    **Attributes**

    :returns results.t (numpy.ndarray): time (ps)
    :returns results.cos_theta_i (numpy.ndarray): Average cos between dipole and axis
    :returns results.cos_theta_ii (numpy.ndarray): Average cos^2 of the same between dipole and axis
    :returns results.cos_theta_ij (numpy.ndarray): Product cos of dipole i and cos of dipole j (i!=j)
    
    """

    def __init__(self,
                 atomgroup,
                 output="dipangle.dat",
                 outfreq=10000,
                 dim=2,
                 **kwargs):
        super().__init__(atomgroup, **kwargs)
        self.output = output
        self.dim = dim
        self.outfreq = outfreq

    def _configure_parser(self, parser):
        parser.description = self.__doc__
        parser.add_argument('-d', dest='dim')
        parser.add_argument('-dout', dest='outfreq')
        parser.add_argument('-o', dest='output')

    def _prepare(self):
        self.n_residues = self.atomgroup.residues.n_residues

        # unit normal vector
        self.unit = np.zeros(3)
        self.unit[self.dim] += 1

        self.cos_theta_i = np.empty(self.n_frames)
        self.cos_theta_ii = np.empty(self.n_frames)
        self.cos_theta_ij = np.empty(self.n_frames)

    def _single_frame(self):

        # make broken molecules whole again!
        self.atomgroup.unwrap(compound="molecules")

        chargepos = self.atomgroup.positions * \
            self.atomgroup.charges[:, np.newaxis]
        dipoles = self.atomgroup.accumulate(
            chargepos, compound=check_compound(self.atomgroup))

        cos_theta = np.dot(dipoles, self.unit) / \
            np.linalg.norm(dipoles, axis=1)
        matrix = np.outer(cos_theta, cos_theta)

        trace = matrix.trace()
        self.cos_theta_i[self._frame_index] = cos_theta.mean()
        self.cos_theta_ii[self._frame_index] = trace / self.n_residues
        self.cos_theta_ij[self._frame_index] = (matrix.sum() - trace)
        self.cos_theta_ij[self._frame_index] /= (self.n_residues**2 -
                                                 self.n_residues)

        if self._save and self._frame_index % self.outfreq == 0 and self._frame_index > 0:
            self._calculate_results()
            self._save_results()

    def _calculate_results(self):
        self._index = self._frame_index + 1

        self.results.t = self._trajectory.dt * \
            np.arange(self.startframe, self.stopframe, self.step)

        self.results.cos_theta_i = self.cos_theta_i[:self._index]
        self.results.cos_theta_ii = self.cos_theta_ii[:self._index]
        self.results.cos_theta_ij = self.cos_theta_ij[:self._index]

    def _save_results(self):

        savetxt(self.output,
                np.vstack([
                    self.results.t, self.results.cos_theta_i,
                    self.results.cos_theta_ii, self.results.cos_theta_ij
                ]).T,
                header="t\t<cos(θ_i)>\t<cos(θ_i)cos(θ_i)>\t<cos(θ_i)cos(θ_j)>",
                fmt='%.5e')


class kinetic_energy(SingleGroupAnalysisBase):
    """Computes the timeseries of energies.

    The kinetic energy function computes the translational and rotational Kinetic Energy for the molecular center of an MD simulation trajectory.

    The program can be run from either a python environment or a command line interface.
    As an example, we analyse a simulation of box of water molecules in an NVE ensemble. 
    The length of the box was 2.5 nm containing around 500 water molecules. 
    The time step of simulation was 4 fs and it was 200 ps long. 
    Periodic boundary conditions were employed in all directions and long range electrostatics were modelled using the PME method. 
    LINCS algorithm was used to constraint the H-Bonds.

    The simulation directory can be downloaded from the repository (shown below) which contains ``nve.tpr`` and ``nve.trr``

    .. code-block:: bash

        cd tests/data/kineticenergy

    **From the python interpreter**

    We need to import `MD Analysis`_, `matplotlib`_ and maicos packages in the environment

    .. code-block:: python3

        import MDAnalysis as mda
        import matplotlib.pyplot as plt
        import maicos

    Then create an MD Analysis Universe. 

    .. code-block:: python3

        u = mda.Universe("nve.tpr","nve.trr")
        at = u.atoms

    Now run the MAICOS dipole angle function.

    .. code-block:: python3

        ke = maicos.dipole_angle(at)
        ke.run()

    The run produces a `python dictionary` named ``ke.results`` with `3 keys` linked to `numpy arrays` as `values`. 
    They are timestep, translational KE, and rotational KE.

    The results can be visualized as follows:

    .. code-block:: python3

        plt.plot(ke.results['t'],ke.results['trans'] )
        plt.title("Translational Kinetic Energy")
        plt.xlabel("Time (ps)")
        plt.ylabel("KJ/mole")
        plt.show()

    The figure generated :

       .. image:: ../images/ket.png
            :width: 600

    MAICOS can also be accessed through command line interface.

    **From the command line interface**

    .. code-block:: bash

        maicos kinetic_energy -s md.tpr -f md.trr -o ke.dat

    The output file ``ke.dat`` is similar to `ke.results` and contains the data in columns. 

    They are several options you can play with. To know the full 
    list of options, have a look at the ``Inputs`` section below. 


    .. _`MD Analysis`: https://www.mdanalysis.org/
    .. _`matplotlib`: https://matplotlib.org/ 


    **Inputs**

    :param output (str): Output filename
    :param refpoint (str): reference point for molecular center: center of
                            mass (COM), center of charge (COC), or oxygen position (OXY)
                            Note: The oxygen position only works for systems of pure water

    **Attributes**

    :returns results.t (numpy.ndarray): time (ps)
    :returns results.trans (numpy.ndarray): translational kinetic energy (kJ/mole)
    :returns results.rot (numpy.ndarray): rotational kinetic energy (kJ/mole)

    """

    def __init__(self, atomgroup, output="ke.dat", refpoint="COM", **kwargs):
        super().__init__(atomgroup, **kwargs)
        self.output = output
        self.refpoint = refpoint

    def _configure_parser(self, parser):
        parser.add_argument('-o', dest='output')
        parser.add_argument('-r', dest='refpoint')

    def _prepare(self):
        """Set things up before the analysis loop begins"""
        if self.refpoint not in ["COM", "COC", "OXY"]:
            raise ValueError(
                "Invalid choice for dens: '{}' (choose from 'COM', "
                "'COC', 'OXY')".format(self.refpoint))

        if self.refpoint == "OXY":
            self.oxy = self.atomgroup.select_atoms("name OW*")

        self.masses = self.atomgroup.atoms.accumulate(
            self.atomgroup.atoms.masses, compound=check_compound(self.atomgroup))
        self.abscharges = self.atomgroup.atoms.accumulate(np.abs(
            self.atomgroup.atoms.charges), compound=check_compound(self.atomgroup))
        # Total kinetic energy
        self.E_kin = np.zeros(self.n_frames)

        # Molecular center energy
        self.E_center = np.zeros(self.n_frames)

    def _single_frame(self):
        self.E_kin[self._frame_index] = np.dot(
            self.atomgroup.masses,
            np.linalg.norm(self.atomgroup.velocities, axis=1)**2)

        if self.refpoint == "COM":
            massvel = self.atomgroup.velocities * \
                self.atomgroup.masses[:, np.newaxis]
            v = self.atomgroup.accumulate(
                massvel, compound=check_compound(self.atomgroup))
            v /= self.masses[:, np.newaxis]

        elif self.refpoint == "COC":
            abschargevel = self.atomgroup.velocities * \
                np.abs(self.atomgroup.charges)[:, np.newaxis]
            v = self.atomgroup.accumulate(
                abschargevel, compound=check_compound(self.atomgroup))
            v /= self.abscharges[:, np.newaxis]

        elif self.refpoint == "OXY":
            v = self.oxy.velocities

        self.E_center[self._frame_index] = np.dot(self.masses,
                                                  np.linalg.norm(v, axis=1)**2)

    def _calculate_results(self):
        self.results.t= self._trajectory.dt * \
            np.arange(self.startframe, self.stopframe, self.step)
        self.results.trans = self.E_center / 2 / 100
        self.results.rot = (self.E_kin - self.E_center) / 2 / 100

    def _save_results(self):
        savetxt(self.output,
                np.vstack([
                    self.results.t, self.results.trans,
                    self.results.rot
                ]).T,
                fmt='%.8e',
                header="t / ps \t E_kin^trans / kJ/mole \t E_kin^rot / kJ/mole")
