.. _pdfs-explanation:

===========================
Pair distribution functions
===========================

The pair distribution function describes the spatial correlation
between particles.

Two-dimensional (planar) pair distribution function
===================================================

Here, we present the two-dimensional pair distribution function
:math:`g_{\text{2d}}(r)`, which restricts the distribution to
particles which lie on the same surface
:math:`S_\xi`.

Let :math:`g_1` be the group of particles which are centered, and :math:`g_2` be
the group of particles whose density around a :math:`g_1` particle is
calculated.
Furthermore, we define a parametric surface :math:`S_\xi` as a function of
:math:`\xi`,

.. math::
   S_\xi = \{ \mathbf{r}_{\xi} (u, v) |
   u_{\text{min}} < u < u_{\text{max}}, v_{\text{min}} < v < v_{\text{max}} \}

which consists of all points :math:`\mathbf{r}_\xi`. By varying
:math:`u, v` we can reach all points on one surface :math:`\xi`. Let us
additionally consider a circle on that plane :math:`S_{i, r}` with radius
:math:`r` around atom :math:`i` given by

.. math::
   S_{i, r} = \{ \mathbf{r}_{i, r} | \; || ( \mathbf{r}_{i, r}
   - \mathbf{x_i} || = r )  \land ( \mathbf{r}_{i, r} \in S_{\xi, i} ) \}

where :math:`S_{\xi, i}` is the plane in which atom :math:`i` lies.

Then the two-dimensional pair distribution function is

.. math::

   g_{\text{2d}}(r) = \left \langle \sum_{i}^{N_{g_1}}
   \frac{1}{L(r, \xi_i)}
   \frac{\sum_{j}^{N_{g_2}} \delta(r - r_{ij}) \delta(\xi_{ij})}
   {\vert \vert \frac{\partial \mathbf{f}_i}{\partial r} \times
   \frac{\partial \mathbf{f}_i}{\partial \xi} \vert \vert _{\phi = \phi_j}}
   \right \rangle

where :math:`L(r, \xi_i)` is the contour length of the circle :math:`S_{i, r}`.
:math:`\mathbf{f}_i(r, \gamma, \phi)` is a parametrization of the
circle :math:`S_{i, r}`.

Discretized for computational purposes we consider a volume
:math:`\Delta V_{\xi_i}(r)`, which is bounded by the surfaces
:math:`S_{\xi_i - \Delta \xi}`, :math:`S_{\xi_i + \Delta \xi}` and
:math:`S_{r - \frac{\Delta r}{2}}, S_{r + \frac{\Delta r}{2}}`. Then our
two-dimensional pair distribution function is

.. math::

   g_{\text{2d}}(r) = \left \langle
   \frac{1}{N_{g_1}} \sum_i^{N_{g_1}}
   \frac{\text{count} \; ({g_2}) \; \text{in} \;\Delta V_{\xi_i}(r)}
   {\Delta V_{\xi_i}(r)}
   \right \rangle

.. _pdfplanar-derivation:


Derivation
----------

Let us introduce cylindrical coordinates :math:`r, z, \phi` with the origin at the
position of atom :math:`i`.

.. math::
   \begin{aligned}
   x &= r \cdot \cos \phi \\
   y &= r \cdot \sin \phi \\
   z &= z \\
   \end{aligned}

Then the two-dimensional pair distribution is given by

.. math::
   g_{\text{2d}}(r, z=0) = \left \langle \sum_{i}^{N_{g_1}}
   \frac{1}{2 \pi r}
   \sum_{j}^{N_{g2}} \delta(r - r_{ij}) \delta(z_{ij})
   \right \rangle

where we have followed the general derivations given above.

For discretized calculation we count the number of atoms per ring as illustrated below

.. image:: ../../static/pdfplanar_sketch.svg
.. image:: ../../static/pdfplanar_sketch.svg
   :alt: Sketch of the discretization
   :class: only-light

.. image:: ../../static/pdfplanar_sketch_dark.svg
.. image:: ../../static/pdfplanar_sketch_dark.svg
   :alt: Sketch of the discretization
   :class: only-dark


The sketch shows an atom :math:`i` from group :math:`g_1`  at the origin in blue.
Around the atom a ring volume with average distance :math:`r` from atom `i`
is shaded in light red.
Atoms :math:`j` from group :math:`g_2` are counted in this volume.

One-dimensional (cylindrical) pair distribution functions
=========================================================

Here, we present the one-dimensional pair distribution functions
:math:`g_{\text{1d}}(\phi)` and :math:`g_{\text{1d}}(z)`, which restricts the
distribution to particles which lie on the same cylinder along the angular and axial
directions respectively.

Let :math:`g2` be the group of particles whose density around a :math:`g1` particle is
to be calculated and let :math:`g1, g2` lie in a cylinderical coordinate
system :math:`(R, z, \phi)`.

Then the angular pair distribution function is

.. math::

   g_{\text{1d}}(\phi) = \left \langle \sum_{i}^{N_{g_1}}
   \sum_{j}^{N_{g2}} \delta(\phi - \phi_{ij}) \delta(R_{ij}) \delta(z_{ij})
   \right \rangle


And the axial pair distribution function is

.. math::

   g_{\text{1d}}(z) = \left \langle \sum_{i}^{N_{g_1}}
   \sum_{j}^{N_{g2}} \delta(z - z_{ij}) \delta(R_{ij}) \delta(\phi_{ij})
   \right \rangle

Discretized for computational purposes we consider a volume
:math:`\Delta V_{z_i,R_i}(\phi)`, which is bounded by the surfaces
:math:`S_{z_i - \Delta z}`, :math:`S_{z_i + \Delta z}`,
:math:`S_{R_i - \Delta R}`, :math:`S_{R_i + \Delta R}` and
:math:`S_{\phi - \frac{\Delta \phi}{2}}, S_{\phi + \frac{\Delta \phi}{2}}`. Then our
the angular pair distribution function is

.. math::

   g_{\text{1d}}(\phi) = \left \langle
   \frac{1}{N_{g_1}} \sum_i^{N_{g_1}}
   \frac{\text{count} \; ({g_2}) \; \text{in} \;\Delta V_{z_i,R_i}(\phi)}
   {\Delta V_{z_i,R_i}(\phi)}
   \right \rangle

Similarly,

.. math::

   g_{\text{1d}}(z) = \left \langle
   \frac{1}{N_{g_1}} \sum_i^{N_{g_1}}
   \frac{\text{count} \; ({g_2}) \; \text{in} \;\Delta V_{\phi_i,R_i}(z)}
   {\Delta V_{\phi_i,R_i}(z)}
   \right \rangle
