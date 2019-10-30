# MAICos - Molecular Analysis for Interfacial and Confined Systems

A Python library to analyse molecular dynamics simulations of
interfacial and confined systems based on
[MDAnalysis](https://www.mdanalysis.org).

# Installation

You'll need [Python3](https://www.python.org) and a C-compiler to build the
underlying libraries. To install the package  
for all users type

```sh
    git clone https://gitlabph.physik.fu-berlin.de/ag-netz/maicos.git
    cd maicos
    pip3 install numpy cython
    pip3 install .
```

To install only on the user site use pip's `--user` flag.
If you are using `BASH` you can add the autocompletion script
by adding

```sh
    source $(maicos --bash_completion)
```

to your `.bashrc` or `.profile` file.

# Usage

You can use maicos either from the command line or directly from your python
code. All available modules are briefly described below.

## From the command line

To run maicos from the command line use

```sh
maicos <module> <paramaters>
```

You can get a help page by typing `maicos -h` or package specific help page
by typing `maicos <package> -h`.

## From the python interpreter

To use maicos with the python interpreter create `analysis` object,
by supplying an MDAnalysis AtomGroup then use the `run` method

```python
import maicos

ana_obj = maicos.<module>(atomgroup, <paramaters>)
ana_obj.run()
```

Results are available through the objects `results` dictionary.

# Modules

Currently `maicos` contains the following analysis modules:

## Density
* **density_planar**: Computes partial densities or temperature profiles across the box.
* **density_cylinder**: Computes partial densities across a cylinder of given radius r and length l

## Dielectric Constant

* **epsilon_bulk**: Computes dipole moment fluctuations and from this the static dielectric constant.
* **epsilon_planar**: Calculates a planar dielectric profile.
* **epsilon_cylinder**: Calculation of the cylindrical dielectric profile for axial (along z) and radial (along xy) direction.
* **dielectric_spectrum**: Calculates the complex dielectric function as a function of the frequency.

## Structure Analysis

* **saxs**: Computes SAXS scattering intensities S(q) for all atom types from the given trajectory.
* **debyer**: Calculates scattering intensities using the debye equation. For using you need to download and build the debyer library see <https://github.com/wojdyr/debyer>.
* **diporder**: Calculates dipolar order parameters

## Timeseries Analysis

* **dipole_angle**: Calculates the timeseries of the dipole moment with respect to an axis.
* **kinetic_energy**: Calculates the timeseries for the molecular center translational and rotational kinetic energy.

## Transport Analysis

* **velocity**: Calculates a velocity profile along a given axis.

# Issues

If you found any bugs, improvements or questions to maicos feel free to raise an
issue.

# Contributing

Contribution via pull requests are always welcome.
For more details see the [README](developer/README.md) in the development section.
