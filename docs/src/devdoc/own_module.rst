Contributing your own analysis module
=====================================

To write your module take a look at the comprehensive example in the documentation of
:class:`maicos.core.AnalysisBase`. MAICoS also has more specific base classes for
different geometries that make developing modules much easier. You may take a look at
the source code at ``src/maicos/modules``.

After you wrote your module you can add it in a new file in ``src/maicos/modules``. On
top of that please also update the list in ``src/maicos/modules/__init__.py``
accordingly. Also, create a new ``.rst`` file with your module name in
``docs/src/references/modules`` similar to the already existing. To finally show the
documentation for the other modules add an entry in
``docs/src/references/modules/index.rst`` in alphabetical order.

All MAICoS modules are also listed in the ``README.rst`` and you should add your module
as well.

Finally, also provide meaningful tests for your module in ``test/modules``.

For further questions feel free to ask us on our Discord_ server.

.. _`Discord`: https://discord.gg/mnrEQWVAed


Observable collection and statistical accumulation in MAICoS
------------------------------------------------------------

When writing own modules, ``MAICoS`` will collect per-frame observables in the private attribute
:attr:`self._obs` object inside your :meth:`_single_frame` method. Each entry in
:attr:`self._obs` should be a numeric scalar or a numpy array. Make sure to keep
the shape of any array-valued observables consistent across frames. (For example
by using the bin numbers provided by the Base classes.)

On the first analysed frame MAICoS initialises several internal accumulators
from :attr:`self._obs`:

- :attr:`self.sums`: running sums of the observables (used for totals or mean accumulation)
- :attr:`self.means`: running means (initialised from the first-frame values)
- :attr:`self.sems`: standard errors of the mean (initially zero or NaN arrays with the correct
  shape)
- :attr:`self.pop`: integer counts per observable/bin
- :attr:`self.M2`: running M2 accumulator used for variance/error estimation

For subsequent frames MAICoS updates these accumulators using a numerically
stable parallel Welford algorithm (see
:func:`maicos.lib.math.combine_subsample_variance`). This merge-based approach supports
vectorised observables and is suitable for combining statistics from separate
blocks of frames.

After merging, the standard error of the mean (SEM) is computed from the
running accumulators and stored in :attr:`self.sems`. The per-observable running sums
are also updated and available in :attr:`self.sums`.

If your :meth:`_single_frame` returns a float, MAICoS collects it into
:attr:`self.timeseries` and later analyses the series for autocorrelation using
:func:`maicos.lib.util.correlation_analysis` during :meth:`_call_conclude`.
The resulting correlation time is stored as :attr:`self.corrtime` and can help you
judge the statistical independence of samples and the reliability of the
estimated :attr:`self.sems`.

Guidelines for observable collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Make sure per-frame observables have a consistent shape across frames.
- Use numeric types (scalars or numpy arrays). Unsupported types will raise a :exc:`TypeError` during the first aggregation step.
- If you compute per-bin averages or use weighting, ensure you also provide a
  matching population/count so the averaging code can correctly compute SEMs.

Following these conventions ensures that MAICoS can provide correct means,
errors and correlation estimates for your analysis module.

