# Basic usage - Command line
# ==========================
#
# .. import
#
# After Installation, any MAICoS module can be accessed simply by using the module name,
# for example in this tutorial we will use :class:`maicos.DensityPlanar`:

maicos densityplanar

# %%
#
# .. loading
#
# We can specify the topology and trajectory using the ``-s`` and ``-f`` flags.

maicos densityplanar -s slit_flow.tpr -f slit_flow.trr

# %%
#
# .. selection
#
# Using the ``-atomgroup`` flag.

maicos densityplanar -s slit_flow.tpr \
                     -f slit_flow.trr \
                     -atomgroup "type OW HW"

# %%
#
# The density profile has been written in a file named ``density.dat`` in the current
# directory. The written file starts with the following lines

head -n 20 density.dat

# %%
#
# .. analysis
#
# For lengthy analysis, use the ``concfreq`` option to update the result during the run

maicos densityplanar -s slit_flow.tpr \
                     -f slit_flow.trr \
                     -atomgroup 'type OW HW' \
                     -concfreq '10'
# %%
#
# .. plot
#
# Using ``gnuplot``:

echo " \
    set xlabel 'z coordinate, (Å)'; \
    set ylabel 'density H2O (u.Å⁻³)'; \
    plot 'density.dat' using (column(1)):(column(2)):(5*column(3)) with yerrorlines title '' \
" | gnuplot || true

# %%
#
# .. help
#
# The general help of MAICoS can be accessed using

maicos -h

# %%
#
# Package-specific page can also be accessed from the cli

maicos densityplanar -h

# %%
#
# .. end
