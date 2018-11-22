# MDTOOLS

A collection of scripts to analyse and build systems for molecular dynamics simulations.
Usually these scripts offer more flexibility and/or are faster than the original implementations.

# Installation

You'll need [Python](https://www.python.org) and a C-compiler to build the
underlying libraries. To install the package  
for all users type
```
git clone https://gitlabph.physik.fu-berlin.de/ag-netz/mdtools.git
cd mdtools
sudo python setup.py install
```
To install it only on the user site use `python setup.py install --user`.
If you are using `BASH` you can add the autocompletion script
by

```
source mdtools-completion.bash
```

To load the completion at every login simply paste the path **given at the end
of the installation** to your `.bashrc` or `.profile` file.

# Usage

You can get a help page by typing `mdtools -h` or package specific help page
by typing `mdtools <package> -h`.


### SAXS using debyer library

For using the debye module you need to download and build
the debyer library see
https://github.com/wojdyr/debyer.

# Contributing

Contribution via pull requests are always welcome. Example code for an
analysis module can be found in the example folder.
