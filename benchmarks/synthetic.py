#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""On-the-fly synthetic data for benchmarks.

Builds a multi-frame :class:`MDAnalysis.Universe` of three-site molecules entirely in
memory (no disk I/O) so that framework and profile benchmarks do not depend on bundled
trajectory files.
"""

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

VOLUME_PER_MOLECULE = 30.0
MASSES = [15.999, 1.008, 1.008]
CHARGES = [-0.8476, 0.4238, 0.4238]


def make_universe(n_atoms: int = 3000, n_frames: int = 10, seed: int = 0):
    """Create an in-memory :class:`MDAnalysis.AtomGroup` for benchmarking.

    Atoms are grouped into net-neutral, water-like molecules of one oxygen bonded to
    two hydrogens sitting 1 Å away in random directions. The bonds are what let MAiCoS
    unwrap and wrap by compound, and the random orientations keep the molecular dipoles
    non-degenerate. ``n_atoms`` is rounded down to a multiple of three and the cubic box
    grows with the system, so that the number density stays that of liquid water.

    Parameters
    ----------
    n_atoms : int
        total number of atoms.
    n_frames : int
        number of trajectory frames.
    seed : int
        seed for the random number generator (deterministic output).
    """
    rng = np.random.default_rng(seed)
    n_molecules = n_atoms // 3
    n_atoms = 3 * n_molecules
    box = (VOLUME_PER_MOLECULE * n_molecules) ** (1 / 3)

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_molecules,
        atom_resindex=np.repeat(np.arange(n_molecules), 3),
        trajectory=True,
    )
    u.add_TopologyAttr("masses", np.tile(MASSES, n_molecules))
    u.add_TopologyAttr("charges", np.tile(CHARGES, n_molecules))
    u.add_TopologyAttr("types", np.tile(["O", "H", "H"], n_molecules))

    oxygens: np.ndarray = np.repeat(3 * np.arange(n_molecules), 2)
    hydrogens = oxygens + np.tile([1, 2], n_molecules)
    u.add_TopologyAttr("bonds", np.column_stack([oxygens, hydrogens]))

    sites = rng.normal(size=(n_frames, n_molecules, 3, 3))
    sites /= np.linalg.norm(sites, axis=-1, keepdims=True)
    sites[..., 0, :] = 0.0
    centers = rng.uniform(0.0, box, size=(n_frames, n_molecules, 1, 3))
    positions = (centers + sites).reshape(n_frames, n_atoms, 3).astype(np.float32)

    u.load_new(
        positions,
        format=MemoryReader,
        velocities=rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32),
        dimensions=np.tile([box, box, box, 90.0, 90.0, 90.0], (n_frames, 1)),
    )
    return u.atoms
