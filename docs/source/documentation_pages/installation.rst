.. _label_installation:

============
Installation
============

`Python3`_ and a C-compiler are needed to build the underlying libraries.

Using pip
---------

If you have root access, install the package for all users by typing in a terminal:

.. code-block:: bash

    pip3 install numpy
    pip3 install maicos

Alternatively, if you don't have special privileges, install the package in your home directory by using the ``--user`` flag:

.. code-block:: bash

    pip3 install --user numpy
    pip3 install --user maicos
    
Bash autocompletion
*******************

You can include MAICoS to ``BASH`` suggestions by oppening your ``.bashrc`` or ``.profile`` file with your favorite text editor (here vim is used):


.. code-block:: bash

    vim ~/.bashrc

and by adding

.. code-block:: bash

    source $(maicos --bash_completion)
    
Development version
-------------------

The development version of MAICoS can be compiled from source. `NumPy`_ and `Cython`_ are required:

.. code-block:: bash

    pip3 install numpy
    pip3 install cython
    
Then type in a terminal:

.. code-block:: bash

    git clone git@gitlab.com:maicos-devel/maicos.git
    pip3 install -e maicos/

Testing
*******

You can run the tests from the ``maicos/tests/`` directory. The tests rely on the `pytest`_ library, and use some work flows from NumPy and `MDAnalysisTests`_. In a terminal, type:

.. code-block:: bash

    pip3 install MDAnalysisTests

Then, type:

.. code-block:: bash

    cd maicos/tests
    pytest  --disable-pytest-warnings
    
.. _`pytest` : https://docs.pytest.org/en/6.2.x/
.. _`NumPy` : https://numpy.org/
.. _`MDAnalysisTests` : https://pypi.org/project/MDAnalysisTests/
.. _`Cython` : https://cython.org/
.. _`Python3`: https://www.python.org

.. toctree::
   :maxdepth: 4
   :hidden:
   :titlesonly:

