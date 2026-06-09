#!/usr/bin/env python
#
# Copyright (c) 2026 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""On-the-fly synthetic data for benchmarks.

Builds a multi-frame :class:`MDAnalysis.Universe` entirely in memory (no disk I/O) so
that framework and profile benchmarks do not depend on bundled trajectory files.
"""

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader


def make_universe(
    n_atoms: int = 3000,
    n_frames: int = 10,
    n_residues: int = 1000,
    box: float = 50.0,
    seed: int = 0,
):
    """Create an in-memory :class:`MDAnalysis.AtomGroup` for benchmarking.

    The universe carries masses, net-neutral per-residue charges, atom types and
    per-frame velocities, and a cubic box of edge length ``box`` for every frame.

    Parameters
    ----------
    n_atoms : int
        total number of atoms.
    n_frames : int
        number of trajectory frames.
    n_residues : int
        number of residues; ``n_atoms`` is distributed evenly across them.
    box : float
        cubic box edge length in Angstrom.
    seed : int
        seed for the random number generator (deterministic output).
    """
    rng = np.random.default_rng(seed)

    resindices = np.repeat(np.arange(n_residues), n_atoms // n_residues)
    # pad the tail so every atom maps to a residue when n_atoms is not divisible
    if resindices.size < n_atoms:
        pad = np.full(n_atoms - resindices.size, n_residues - 1)
        resindices = np.concatenate([resindices, pad])

    u = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=resindices,
        trajectory=True,
    )
    u.add_TopologyAttr("masses", rng.uniform(1.0, 16.0, n_atoms))
    # net-neutral charges per residue: one atom balances the rest (e.g. water-like).
    # maicos checks neutrality per compound (here residues) for dielectric analyses.
    counts = np.bincount(resindices, minlength=n_residues)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    charges = np.full(n_atoms, 0.5)
    charges[first] = -0.5 * (counts - 1)
    u.add_TopologyAttr("charges", charges)
    u.add_TopologyAttr("types", ["O"] * n_atoms)

    positions = rng.uniform(0.0, box, size=(n_frames, n_atoms, 3)).astype(np.float32)
    velocities = rng.normal(size=(n_frames, n_atoms, 3)).astype(np.float32)
    dimensions = np.tile([box, box, box, 90.0, 90.0, 90.0], (n_frames, 1))

    u.load_new(
        positions,
        format=MemoryReader,
        velocities=velocities,
        dimensions=dimensions,
    )
    return u.atoms
