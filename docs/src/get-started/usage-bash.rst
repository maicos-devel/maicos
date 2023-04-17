Usage - command line
====================

MAICoS can be used directly from the command line (cli). Using cli instead of a Jupyter
notebook can sometimes be more comfortable, particularly for lengthy analysis. The cli
in particular is handy because it allows for updating the analysis results during the
run. You can specify the number of frames after the output is updated with the
``-concfreq`` flag. See below for details.

Note that in this documentation, we almost exclusively describe the use of MAICoS from
the python interpreter, but all operations can be equivalently performed from the cli.

.. literalinclude:: ../../../examples/usage-bash.sh
    :language: bash
