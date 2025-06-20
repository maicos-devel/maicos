.. _saxs-explanations:

============================
Small-angle X-ray scattering
============================

MD Simulations often complement conventional experiments, such as X-ray crystallography,
Nuclear Magnetic Resonance (NMR) spectroscopy and Atomic-Force Microscopy (AFM). X-ray
crystallography is a method by which the structure of molecules can be resolved. X-rays
of wavelength 0.1 to 100 Å are scattered by the electrons of atoms. The intensities of
the scattered rays are often amplified using by crystals containing a multitude of the
studied molecule positionally ordered. The molecule is thereby no longer under
physiological conditions. However, the study of structures in a solvent should be done
under physiological conditions (in essence, this implies a disordered, solvated fluid
system); therefore X-ray crystallography does not represent the ideal method for such
systems. Small-Angle X-ray Scattering (abbreviated to SAXS) allows for measurements of
molecules in solutions. With this method the shape and size of the molecule and also
distances within it can be obtained. In general, for larger objects, the information
provided by SAXS can be converted to information about the object's geometry via the
Bragg-Equation

.. math::
    n \cdot \lambda = 2 \cdot d \cdot \sin(\theta)

with :math:`n \in \mathbb{N}`, :math:`\lambda` the wavelength of the incident wave,
:math:`d` the size of the diffracting object, and :math:`\theta` the scattering angle.
For small angles, :math:`d` and :math:`\theta` are approximately inversely proportional
to each other, which means larger objects scatter X-rays at smaller angles.

-----------
Experiments
-----------

The measured quantity in SAXS experiments is the number of elastically scattered photons
as a function of the scattering angle :math:`2\theta`, i.e. the intensity of the
scattered rays across a range of small angles. The general set-up of a SAXS experiment
is shown in figure below.

.. image:: ../../static/saxs-light.png
   :alt: Setup of a SAXS
   :class: only-light

.. image:: ../../static/saxs-dark.png
   :alt: Setup of a SAXS
   :class: only-dark

The experiments are carried out by placing the sample of interest in a highly
monochromatic and collimated (parallel) X-ray beam of wavelength :math:`\lambda`. When
the incident rays with wave vector :math:`\boldsymbol{k}_i` reach the sample they
scatter. The scattered rays, with wave vector :math:`\boldsymbol{k}_s`, are recorded by
a 2D-detector revealing a diffraction pattern.

Since the scattering agents in the sample are electrons, X-Ray diffraction patterns
reveal the electron density. Since the scattering is elastic, the magnitudes of the
incident and scattered waves are the same: :math:`|\boldsymbol{k}_i| =
|\boldsymbol{k}_s| = 2\pi/\lambda`. The scattering vector is :math:`\boldsymbol{q} =
\boldsymbol{k}_s - \boldsymbol{k}_i` with a magnitude of :math:`q = |\boldsymbol{q}| =
4\pi \sin(\theta)/\lambda`. The structure factor can be obtained from the intensity of
the scattered wave, :math:`I_s(\boldsymbol{q})`, and the correspnding atomic form factor
:math:`f (q)`, which involves a frourier transform of the element-specific local
electron density and thus determines the amplitude of the scattered wave of a single
element.

-----------
Simulations
-----------

In simulations, the structure factor and scattering intensities
:math:`S(\boldsymbol{q})` can be extracted directly from the positions of the particles.
:class:`maicos.Saxs` calculates these factors. The calculated scattering intensities can
be directly compared to the experimental one without any further processing. In the
following we derive the essential relations. We start with the scattering intensity
which is expressed as

.. math::
    I_s(\boldsymbol{q}) = A_s(\boldsymbol{q}) \cdot A_s^*(\boldsymbol{q}) \,,

with the amplitude of the elastically scattered wave

.. math::
    A_s(\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) \cdot e^{-i\boldsymbol{qr}_j} \,,

where :math:`f_j(q)` is the element-specific atomic form factor of atom :math:`j` and
:math:`\boldsymbol{r}_j` the position of the :math:`j` th atom out of :math:`N` atoms.

The scattering intensity can be evaluated for wave vectors :math:`\boldsymbol q = 2 \pi
(L_x n_x, L_y n_y, L_z n_z)`, where :math:`n \in \mathbb N` and :math:`L_x, L_y, L_z`
are the box lengths of cubic cells.

.. Note::
    :class:`maicos.Saxs` can analyze any cells by mapping coordinates back onto cubic
    cells.

The complex conjugate of the amplitude is

.. math::
    A_s^*(\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) \cdot e^{i\boldsymbol{qr}_j} \,.

The scattering intensity therefore can be written as

.. math::
    I_s (\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) e^{-i\boldsymbol{qr}_j}
                            \cdot \sum\limits_{k=1}^N f_k(q) e^{i\boldsymbol{qr}_k} \,.

With Euler’s formula :math:`e^{i\phi} = \cos(\phi) + i \sin(\phi)` the intensity is

.. math::
    I_s (\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) \cos(\boldsymbol{qr}_j) - i \sin(\boldsymbol{qr}_j)
                            \cdot \sum\limits_{k=1}^N f_k(q) \cos(\boldsymbol{qr}_k) + i \sin(\boldsymbol{qr}_k) \,.

Multiplication of the terms and simplifying yields the final expression for the
intensity of a scattered wave as a function of the wave vector and with respect to the
particle’s atomic form factor

.. math::
    I_s (\boldsymbol{q}) = \left[ \sum\limits_{j=1}^N f_j(q) \cos(\boldsymbol{qr}_j) \right ]^2 +
                           \left[ \sum\limits_{j=1}^N f_j(q) \sin(\boldsymbol{qr}_j) \right ]^2 \,.

For systems containing only one kind of atom the structure factor is connected to the
scattering intensity via

.. math::
    I_s (\boldsymbol{q}) = [f(q)]^2 S(\boldsymbol{q}) \,.

For any system the structure factor can be written as

.. math::
    S(\boldsymbol{q}) =
        \left\langle \frac{1}{N}\sum\limits_{j=1}^N \cos(\boldsymbol{qr}_j) \right \rangle^2 +
        \left\langle \frac{1}{N} \sum\limits_{j=1}^N \sin(\boldsymbol{qr}_j) \right \rangle^2 \,.


The limiting value :math:`S(0)` for :math:`q \rightarrow 0` is connected to the
isothermal compressibility :footcite:p:`hansen_theory_2006` and the element-specific
atomic form factors :math:`f(q)` of a specific atom can be approximated with

.. math::
    f(\sin\theta/\lambda) = \sum_{i=1}^4 a_i e^{-b_i \sin^2\theta/\lambda^2} + c \,.

Expressed in terms of the scattering vector we can write

.. math::
    f(q) = \sum_{i=1}^4 a_i e^{-b_i q^2/(4\pi)^2} + c \,.

The element-specific coefficients :math:`a_{1,\dots,4}`, :math:`b_{1,\dots,4}` and
:math:`c` are documented :footcite:p:`princeInternationalTablesCrystallography2004`.

.. attention::

    The atomic form factor should not be confused with the atomic scattering factor or
    intensity (often anonymously called form factor). The scattering intensity depends
    strongly on the distribution of atoms and can be computed using
    :class:`maicos.Saxs`.

----------------------------------------------------------------------
Connection of the structure factor to the radial distribution function
----------------------------------------------------------------------

If the system's structure is determined by pairwise interactions only, the density
correlations of a fluid are characterized by the pair distribution function

.. math::
    g(\boldsymbol r, \boldsymbol r^\prime) =
        \frac{\langle \rho^{(2)}(\boldsymbol r, \boldsymbol r^\prime) \rangle}
        {\langle \rho(\boldsymbol r) \rangle \langle \rho(\boldsymbol r\prime) \rangle}
    \,,

where :math:`\rho^{(2)}(\boldsymbol r, \boldsymbol r\prime) = \sum_{i,j=1, i\neq j}^{N}
\delta (\boldsymbol r - \boldsymbol r_i) \delta (\boldsymbol r - \boldsymbol r_j)` and
:math:`\rho(\boldsymbol r) = \sum_{i=1}^{N} \delta (\boldsymbol r - \boldsymbol r_i)`
are the two- and one-particle density operators.

For a homogeneous and isotropic system, :math:`g(r) = g(\boldsymbol r, \boldsymbol
r^\prime)` is a function of the distance :math:`r =|\boldsymbol r - \boldsymbol
r^\prime|` only and is called the radial distribution function (RDF). As explained
above, scattering experiments measure the structure factor

.. math::
    S(\boldsymbol q) = \left \langle \frac{1}{N} \sum_{i,j=1}^N
        \exp(-i\boldsymbol q \cdot [\boldsymbol r_i - \boldsymbol r_j]) \right \rangle
    \,,

which we here normalize only by the number of particles :math:`N`. For a homogeneous and
isotropic system, it is a function of :math:`q = |\boldsymbol q|` only and related to
the RDF by Fourier transformation (FT)

.. math::
    S^{FT}(q) = 1 + 4 \pi \rho \int_0^\infty \mathrm{d}r r \frac{\sin(qr)}{q} (g(r) - 1) \,,

which is another way compared for the direct evaluation from trajectories which was
derived above. In general this can be as accurate as the direct evaluation if the
RDF implementation works for non-cubic cells and is not limited to distances
:math:`r_\mathrm{max} = L/2`, see :footcite:p:`zeman_ionic_2021` for details.
However, in usual implementation the RDF can only be obtained until
:math:`r_\mathrm{max} = L/2` which leads to a range of :math:`q >
q_\mathrm{min}^\mathrm{FT} = 2\pi / r_\mathrm{rmax} = 4 \pi /L`. This means that the
minimal wave vector that can be resolved is a factor of 2 larger compared compared to
the direct evaluation, leading to "cutoff ripples". The direct evaluation should
therefore usually be preferred :footcite:p:`sedlmeier_spatial_2011`.

To compare the RDF and the structure factor you can use
:func:`maicos.lib.math.rdf_structure_factor`. For a detailed example take
a look at :ref:`howto-saxs`.

References
----------
.. footbibliography::
