#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import copy

import MDAnalysis
import numpy as np

def box(TargetUniverse, ProjectileUniverse, InsertionShift=None, zmin=0, zmax=None, distance=1.25):
    """Inserts the Projectile atoms into a box in the TargetUniverse
    at random position and orientation and returns a new Universe."""

    if zmax == None:
        zmax = TargetUniverse.dimensions[2]

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = np.copy(TargetUniverse.dimensions)
    cogProjectile = ProjectileUniverse.atoms.center_of_geometry()

    if InsertionShift == None:
        InsertionDomain = np.array(
            (TargetUniverse.dimensions[0], TargetUniverse.dimensions[1], (zmax - zmin)))

    TargetUniverse = MDAnalysis.Merge(
        TargetUniverse.atoms, ProjectileUniverse.atoms)

    target = TargetUniverse.atoms[0:nAtomsTarget]
    projectile = TargetUniverse.atoms[-nAtomsProjectile:]

    projectile.translate(-cogProjectile)
    initialpositionsProjectile = projectile.positions

    ns = MDAnalysis.lib.NeighborSearch.AtomNeighborSearch(target)

    for attempt in range(1000):   # Generate coordinates and check for overlap

        newangles = np.random.rand(3) * 360
        projectile.rotateby(newangles[0], [1, 0, 0])
        projectile.rotateby(newangles[1], [0, 1, 0])
        projectile.rotateby(newangles[2], [0, 0, 1])

        newcoord = np.random.rand(3) * InsertionDomain
        newcoord[2] += zmin
        projectile.translate(newcoord)

        if len(ns.search(projectile, distance)) == 0:
            break

        projectile.positions = initialpositionsProjectile
    else:
        print("No suitable position found")

    TargetUniverse.dimensions = dimensionsTarget
    projectile.residues.resids = projectile.residues.resids + \
        target.residues.resids[-1]

    return TargetUniverse


def cyzone(TargetUniverse, ProjectileUniverse, radius,
           InsertionShift=np.array([0, 0, 0]), zmin=0, zmax=None):
    """Insert the Projectile atoms into a cylindrical zone in the
    around TargetUniverse's center of geometry at random position and orientation
    and returns a new Universe."""

    if zmax == None:
        zmax = TargetUniverse.dimensions[2]

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = np.copy(TargetUniverse.dimensions)
    cogProjectile = ProjectileUniverse.atoms.center_of_geometry()
    cogTarget = TargetUniverse.atoms.center_of_geometry()

    if InsertionShift.any() == np.array([0, 0, 0]).any():
        InsertionShift = np.array((cogTarget[0], cogTarget[1], zmin))

    TargetUniverse = MDAnalysis.Merge(
        TargetUniverse.atoms, ProjectileUniverse.atoms)

    target = TargetUniverse.atoms[0:nAtomsTarget]
    projectile = TargetUniverse.atoms[-nAtomsProjectile:]

    projectile.translate(-cogProjectile)
    initialpositionsProjectile = projectile.positions

    ns = MDAnalysis.lib.NeighborSearch.AtomNeighborSearch(target)

    for attempt in range(1000):   # Generate coordinates and check for overlap

        newangles = np.random.rand(3) * 360
        projectile.rotateby(newangles[0], [1, 0, 0])
        projectile.rotateby(newangles[1], [0, 1, 0])
        projectile.rotateby(newangles[2], [0, 0, 1])

        newr = np.random.rand() * radius
        newphi = np.random.rand() * 2 * np.pi
        newz = np.random.rand() * (zmax - zmin)

        newcoord = np.array(
            [newr * np.cos(newphi), newr * np.sin(newphi), newz])
        newcoord += InsertionShift
        newcoord[2] += zmin
        projectile.translate(newcoord)

        if len(ns.search(projectile, 1.25)) == 0:
            break

        projectile.positions = initialpositionsProjectile
    else:
        print("No suitable position found")

    TargetUniverse.dimensions = dimensionsTarget
    projectile.residues.resids = projectile.residues.resids + \
        target.residues.resids[-1]

    return TargetUniverse


def repairMolecules(selection):
    """Repairs molecules that are broken due to peridodic boundaries.
    To this end the center of mass is reset into the central box.
    CAVE: Only works with small (< half box) molecules."""

    # we repair each moleculetype individually for performance reasons
    for seg in selection.segments:
        atomsPerMolecule = seg.atoms.n_atoms // seg.atoms.n_residues

        # Make molecules whole, use first atom as reference
        distToFirst = np.empty((seg.atoms.positions.shape))
        for i in range(atomsPerMolecule):
            distToFirst[i::atomsPerMolecule] = seg.atoms.positions[i::atomsPerMolecule] - \
                seg.atoms.positions[0::atomsPerMolecule]
        seg.atoms.positions -= (np.abs(distToFirst) >
                                selection.dimensions[:3] / 2.) * selection.dimensions[:3] * np.sign(distToFirst)

        # Calculate the centers of the objects ( i.e. Molecules )
        masspos = (seg.atoms.positions * seg.atoms.masses[:, np.newaxis]).reshape(
            (seg.atoms.n_atoms // atomsPerMolecule, atomsPerMolecule, 3))
        # all molecules should have same mass
        centers = np.sum(masspos.T, axis=1).T / \
            seg.atoms.masses[:atomsPerMolecule].sum()

        # now shift them back into the primary simulation cell
        seg.atoms.positions += np.repeat(
            (centers % selection.dimensions[:3]) - centers, atomsPerMolecule, axis=0)
