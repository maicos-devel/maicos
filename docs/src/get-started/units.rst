Some conventions
----------------

The base units of MAICoS are consistent with those of `MDAnalysis`_.
Keeping inputs and outputs consistent with this set of units reduces ambiguity,
so we encourage everyone to use them exclusively.

.. _`MDAnalysis` : https://docs.mdanalysis.org/stable/documentation_pages/units.html

The base units are:

.. Table:: Base units in MDAnalysis

   =========== ============== ===============================================
   quantity    unit            SI units
   =========== ============== ===============================================
   length       Å              :math:`10^{-10}` m
   mass         u              :math:`1.660538921 \times 10^{-27}` kg
   time         ps             :math:`10^{-12}` s
   energy       kJ/mol         :math:`1.66053892103219 \times 10^{-21}` J
   charge       :math:`e`      :math:`1.602176565 \times 10^{-19}` As
   force        kJ/(mol·Å)     :math:`1.66053892103219 \times 10^{-11}` J/m
   speed        Å/ps           :math:`100` m/s
   =========== ============== ===============================================
