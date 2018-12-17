# MDTOOLS

A collection of scripts to analyse molecular dynamics simulations.
Usually these scripts offer more flexibility and/or are faster than the original implementations.

# Installation

You'll need [Python](https://www.python.org) and a C-compiler to build the
underlying libraries. To install the package  
for all users type

```sh
    git clone https://gitlabph.physik.fu-berlin.de/ag-netz/mdtools.git
    cd mdtools
    sudo python setup.py install
```

To install it only on the user site use `python setup.py install --user`.
If you are using `BASH` you can add the autocompletion script
by adding

```sh
    source $(mdtools --bash_completion)
```

to your `.bashrc` or `.profile` file.

# Usage

You can use mdtools either from the command line or directly from your python
code. All available modules are briefly described below.

## From the command line

To run mdtools from the command line use

```sh
mdtools <package> <paramaters>
```

You can get a help page by typing `mdtools -h` or package specific help page
by typing `mdtools <package> -h`.

## From the python interpreter

To use mdtools with the python interpreter create `analysis` object,
by supplying an MDAnalysis AtomGroup then use the `run` method

```python
import mdtools.ana

ana_obj = mdtools.ana.<metamodule>.<module>(atomgroup, <paramaters>)
ana_obj.run()
```

Results are available through the objects `results` dictionary.

# Modules

## Density

### density_planar

Computes partial densities or temperature profiles across the box.

## Epsilon

### epsilon_bulk

Computes the dipole moment flcutuations and from this thedielectric constant.

### epsilon_planar

Calculate the dielectric profile.
See Bonthuis et. al., Langmuir 28, vol. 20 (2012) and
Schlaich et. al., PRL 4, vol. 117 (2016) for details.

### dielectric_spectrum

Calculates the complex dielectric function as a function of the frequency.

## Structure Analysis

### saxs

Computes SAXS scattering intensities S(q) for all atom types from the given trajectory.

### debyer

Calculates scattering intensities using the debye equation.
For using the debye module you need to download and build
the debyer library see <https://github.com/wojdyr/debyer>.

## Timeseries

### dipole_angle

Calculates the timeseries of the dipole moment with respect to an axis.

### kinetic_energy

Calculates the timeseries for the molecular center translational
and rotational kinetic energy.

# Issues

If you found any bugs, improvements or questions to mdtools feel free to raise an
issue.

# Contributing

Contribution via pull requests are always welcome.
For more details see the [README](developer/README.md) in the development section.
