# Usage - command line
# ====================
#
# MAICoS can be used directly from the command line (cli). Using cli instead of a Jupyter
# notebook can sometimes be more comfortable, particularly for lengthy analysis. The cli
# in particular is handy because it allows for updating the analysis results during the
# run. You can specify the number of frames after the output is updated with the
# ``-concfreq`` flag. See below for details.
#
# Note that in this documentation, we almost exclusively describe the use of MAICoS from
# the python interpreter, but all operations can be equivalently performed from the cli.

maicos densityplanar -s slit_flow.tpr \
                     -f slit_flow.trr \
                     -atomgroup 'type OW HW'

# %%
#
# The density profile has been written in a file named ``density.dat`` in the current
# directory. The written file starts with the following lines

head -n 20 density.dat

# %%
#
# For lengthy analysis, use the ``concfreq`` option to update the result during the run

maicos densityplanar -s slit_flow.tpr \
                    -f slit_flow.trr \
                    -atomgroup 'type OW HW' \
                    -concfreq '10'

# %%
#
# The general help of MAICoS can be accessed using

maicos -h

# %%
#
# Package-specific page can also be accessed from the cli

maicos densityplanar -h
