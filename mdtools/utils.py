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

dt_dk_tolerance = 1e-8 # Max variation from the mean dt or dk that is allowed (~1e-10 suggested)

def FT(t, x, indvar=True):
    """Discrete fast fourier transform.\
    Takes the time series and the function as arguments.\
    By default, returns the FT and the frequency:\
    setting indvar=False means the function returns only the FT."""
    a, b = np.min(t), np.max(t)
    dt = (t[-1] - t[0])/float( len(t)-1 ) # timestep
    if (abs((t[1:]-t[:-1] - dt)) > dt_dk_tolerance).any():
        print(np.max( abs(t[1:]-t[:-1])))
        raise RuntimeError("Time series not equally spaced!")
    N = len(t)
    # calculate frequency values for FT
    k = np.fft.fftshift(np.fft.fftfreq(N,d=dt)*2*np.pi)
    # calculate FT of data
    xf = np.fft.fftshift(np.fft.fft(x))
    xf2 = xf*(b-a)/N*np.exp(-1j*k*a)
    if indvar:
        return k, xf2
    else:
        return xf2

def iFT(k, xf, indvar=True):
    """Inverse discrete fast fourier transform.\
    Takes the frequency series and the function as arguments.\
    By default, returns the iFT and the time series:\
    setting indvar=False means the function returns only the iFT."""
    dk = (k[-1] - k[0])/float( len(k)-1 ) # timestep
    if (abs((k[1:]-k[:-1] - dk)) > dt_dk_tolerance).any():
        print(np.max( abs(k[1:]-k[:-1])))
        raise RuntimeError("Time series not equally spaced!")
    N = len(k)
    x = np.fft.ifftshift(np.fft.ifft(xf))
    t = np.fft.ifftshift(np.fft.fftfreq(N, d=dk))*2*np.pi
    if N%2 == 0:
        x2 = x*np.exp(-1j*t*N*dk/2.)*N*dk/(2*np.pi)
    else:
        x2 = x*np.exp(-1j*t*(N-1)*dk/2.)*N*dk/(2*np.pi)
    if indvar:
        return t, x2
    else:
        return x2

def Correlation(a, b=None, subtract_mean=False):
    """Uses fast fourier transforms to give the correlation function\
    of two arrays, or, if only one array is given, the autocorrelation.\
    Setting subtract_mean=True causes the mean to be subtracted from the input data."""
    meana = int(subtract_mean)*np.mean(a) # essentially an if statement for subtracting mean
    a2 = np.append( a-meana, np.zeros(2**int( np.ceil( (np.log(len(a))/np.log(2)) ) ) - len(a)) ) # round up to a power of 2
    data_a = np.append( a2, np.zeros(len(a2)) ) # pad with an equal number of zeros
    fra = np.fft.fft(data_a) # FT the data
    if b is None:
        sf = np.conj(fra)*fra # take the conj and multiply pointwise if autocorrelation
    else:
        meanb = int(subtract_mean)*np.mean(b)
        b2 = np.append( b-meanb, np.zeros( 2**int( np.ceil( ( np.log(len(b))/np.log(2)) ) )-len(b) ) )
        data_b = np.append(b2, np.zeros(len(b2)))
        frb = np.fft.fft( data_b )
        sf = np.conj(fra)*frb
    cor = np.real(np.fft.ifft(sf)[:len(a)])/np.array( range(len(a),0,-1) ) # inverse FFT and normalization
    return cor

def ScalarProdCorr(a, b=None, subtract_mean=False):
    """Gives the correlation function of the scalar product of two vector timeseries.\
    Arguments should be given in the form a[t, i], where t is the time variable,\
    along which the correlation is calculated, and i indexes the vector components."""
    corr = np.zeros(len(a[:,0]))

    if b is None:
        for i in range(0, len(a[0,:])):
            corr[:] += Correlation(a[:,i], None, subtract_mean)

    else:
        for i in range(0, len(a[0,:])):
            corr[:] += Correlation(a[:,i], b[:,i], subtract_mean)

    return corr
    
    
def nlambdas(mdp):
    """Returns the total number of lambda states of a given MDP dictionary"""
    for t in "fep", "coul", "vdw", "bonded", "restraint", "mass", "temperature":
        try:
            return len(mdp['{}-lambdas'.format(t)])
        except KeyError:
            pass
    return 0


def replace(file_path, pattern, subst, new_file_path=None):
    """Replaces the pattern with subst in the given file by file path
    and writes it to new_file_path."""
    with open(file_path, 'r') as f:
        read_data = f.read()
    if new_file_path == None:
        new_file_path = file_path
    with open(new_file_path, "w") as f:
        f.write(read_data.replace(str(pattern), str(subst)))


def copy_itp(path_to_topfile, new_path="."):
    """Copy all non standard from the path_to_topfile to new_path."""
    topdir = os.path.dirname(path_to_topfile)
    itp_files = [itp for itp in os.listdir(topdir) if ".itp" in itp]
    for file_name in itp_files:
        shutil.copy("{}/{}".format(topdir, file_name), new_path)


def submit_job(subfile_path, jobname, command):
    """Appends the commands to the subfile given by subfile_path, sets the jobname and submits the jib using SLURM."""
    with open(subfile_path) as f:
        submission_file = f.read()

    submission_file = submission_file.replace('job_name', jobname)
    submission_file += "{}\n".format(command)

    with open('srun.sh', 'w') as f:
        f.write(submission_file)

    subprocess.call('sbatch srun.sh', shell=True)
