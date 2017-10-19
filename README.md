# MDANA
A collection of useful tools and snipplets for trajectory analysis in MDAnalysis. To use the repositry clone the 
repository and __submodules__

```bash
git clone --recursive https://gitlabph.physik.fu-berlin.de/ag-netz/mdana.git
```

## Useful replacements for standard analysis

Usually these scripts offer more flexibility and/or are faster than the original implementations

### density

Computes partial densities across the box.


### insert

Inserts molecules at random positions wir random orientations in a slab geometry.


### SAXS using debyer library

Computes the scattering intensities using the Debye scattering equation
(see https://en.wikipedia.org/wiki/Structure_factor for details). The
atomic form factors according to a Cromer-Mann fit.
(see International Tables for Crystallography,
  Volume C: Mathematical, Physical and Chemical Tables page 554.)

Before using it you have to compile the underlying debyer library see 

```python
python3 setup-sfactor.py build_ext --inplace
```

The default compiler is `gcc-7`. You can adjust the compiler by changing the value of
`os.environ["CC"]` in the `setup-sfactor.py` file. If you want to use the default compiler of your
system just comment this line out.
