.. _mathematical_helper_functions:

Mathematical helper functions
#############################

.. automodule:: maicos.lib.math
    :members:
    :undoc-members:
    :show-inheritance:

.. automodule:: maicos.lib._cmath
    :members:
    :undoc-members:
    :show-inheritance:


Correlation time
----------------

.. autofunction:: maicos.lib.math.correlation_time

   **Example**

   Estimate the correlation time of a correlated noise time series:

   .. code-block:: python

      import numpy as np
      from maicos.lib.math import correlation_time

      np.random.seed(0)

      dt = 0.1
      tau_true = 5.0
      n = 1000

      noise = np.random.normal(size=n)
      signal = np.zeros(n)

      alpha = np.exp(-dt / tau_true)

      for i in range(1, n):
          signal[i] = alp
