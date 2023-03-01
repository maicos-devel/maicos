.. _saxs-explanations:

================
SAXS calculation
================

MD Simulations often complement conventional experiments, such as X-ray
crystallography, Nuclear Magnetic Resonance (NMR) spectroscopy
and Atomic-Force Microscopy (AFM). X-ray crystallography is a method by
which the structure of molecules can be resolved. X-rays of wavelength
0.1 to 100 Å are scattered by the electrons of atoms. The intensities
of the scattered rays are amplified by creating crystals containing a
multitude of the studied molecule positionally ordered. The molecule
is thereby no longer under physiological conditions. The study of
structures in a solvent should be done under physiological conditions
(in essence a disordered system), wherefore X-ray crystallography does not
represent the ideal method. Small-Angle X-ray Scattering (abbreviated to SAXS)
allows for measurements to be made on molecules in solutions. With this method
the shape and size of the molecule and also distances within it can be
obtained. For generally larger objects, the information provided by SAXS can be converted to information
about the objects geometry from the Bragg-Equation

.. math::
    n \cdot \lambda = 2 \cdot d \cdot \sin(\theta)

with :math:`n \in \mathbb{N}`, :math:`\lambda` the wavelength of the incident
wave, :math:`d` the size of the diffracting object, and 
:math:`\theta` the scattering angle. For small angles, :math:`d` and :math:`\theta` 
are approximately inversely proportional to each other, which means larger objects scatter X-rays at smaller angles.

-----------
Experiments
-----------

The measured quantity in SAXS experiments is the number of elastically
scattered photons as a function of the scattering angle :math:`2\theta`, i.e.
the intensity of the scattered rays across a range of small angles.
The general set-up of a SAXS experiment is shown in figure below.

.. image:: ../../static/saxs-light.png
   :alt: Setup of a SAXS
   :class: only-light

.. image:: ../../static/saxs-dark.png
   :alt: Setup of a SAXS
   :class: only-dark

The experiments are carried out by placing the sample of interest in a highly 
monochromatic and collimated (parallel) X-ray beam of wavelength :math:`\lambda`.
When the incident rays with wave vector :math:`\boldsymbol{k}_i` reach the
sample they scatter. The scattered rays, with wave vector :math:`\boldsymbol{k}_s`, are recorded by
a 2D-detector revealing a diffraction pattern.

The scattering agents in the sample are electrons. Therefore diffraction patterns
reveal the electron density. Because the scattering is elastic, the
magnitudes of the incident and scattered waves are the same:
:math:`|\boldsymbol{k}_i| = |\boldsymbol{k}_s| = 2\pi/\lambda`.
The scattering vector is :math:`\boldsymbol{q} = \boldsymbol{k}_s - \boldsymbol{k}_i`
with a magnitude of :math:`q = |\boldsymbol{q}| = 4\pi \sin(\theta)/\lambda`.
The structure factor can be obtained from the intensity of the scattered wave, :math:`I_s(\boldsymbol{q})`, 
and each particle`s form factor :math:`f (q)`. 

-----------
Simulations
-----------

In simulations, the structure factor 
:math:`S(\boldsymbol{q})` can be extracted directly from the positions of 
the particles. MAICoS' :ref:`Saxs` module calculates these factors.
The calculated 
scattering intensities can be directly compared to the experimental one without
any further processing. We now derive the essential equations. 
:math:`S(\boldsymbol{q})` is defined as

.. math::
    S(\boldsymbol{q}) = \frac{1}{\sum_{j=1}^N f_j^2(q)} I_s(\boldsymbol{q}) \,.

The form factor as a function of :math:`q` is specific to each atom and relates
to the amplitude of the scattered waves.

The scattering intensity is expressed as

.. math::
    I_s(\boldsymbol{q}) = A_s(\boldsymbol{q}) \cdot A_s^*(\boldsymbol{q}) \,,

with the amplitude of the elastically scattered wave

.. math::
    A_s(\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) \cdot e^{-i\boldsymbol{qr}_j} \,,

:math:`f_j(q)` is the form factor and :math:`\boldsymbol{r}_j` the position of
the :math:`j` th atom out of :math:`N` atoms. The complex conjugate of the amplitude is

.. math::
    A_s^*(\boldsymbol{q}) = \sum\limits_{k=1}^N f_k(q) \cdot e^{i\boldsymbol{qr}_j} \,.

The intensity therefore can be written as

.. math::
    I_s (\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) e^{-i\boldsymbol{qr}_j}
                            \cdot \sum\limits_{k=1}^N f_k(q) e^{i\boldsymbol{qr}_k} \,.

With Euler’s formula :math:`e^{i\phi} = \cos(\phi) + i \sin(\phi)` 
the intensity is

.. math::
    I_s (\boldsymbol{q}) = \sum\limits_{j=1}^N f_j(q) \cos(\boldsymbol{qr}_j) - i \sin(\boldsymbol{qr}_j)
                            \cdot \sum\limits_{k=1}^N f_k(q) \cos(\boldsymbol{qr}_k) + i \sin(\boldsymbol{qr}_k) \,.

Multiplication of the terms and simplifying yields the final expression
for the intensity of a scattered wave as a function of the wave vector
and with respect to the particle’s form factor

.. math::
    I_s (\boldsymbol{q}) = \left[ \sum\limits_{j=1}^N f_j(q) \cos(\boldsymbol{qr}_j) \right ]^2 +
                           \left[ \sum\limits_{j=1}^N f_j(q) \sin(\boldsymbol{qr}_j) \right ]^2 \,.

For isotropic systems containing only one kind of atom the structure factor is

.. math::
    S(\boldsymbol{q}) = \left\langle \frac{1}{N}\sum\limits_{j=1}^N f_j(q) \cos(\boldsymbol{qr}_j) \right \rangle^2 +
                        \left\langle \frac{1}{N} \sum\limits_{j=1}^N f_j(q) \sin(\boldsymbol{qr}_j) \right \rangle^2 \,.

The structure factor of systems with more than one atom type is the sum of
partial structure factors normalised by the form factor

.. math::
    S(\boldsymbol{q}) = \left\langle \frac{1}{\sum_{j=1}^N f_j^2(q)}\sum\limits_{j=1}^N f_j(q) \cos(\boldsymbol{qr}_j) \right \rangle^2 +
                        \left\langle \frac{1}{\sum_{j=1}^N f_j^2(q)} \sum\limits_{j=1}^N f_j(q) \sin(\boldsymbol{qr}_j) \right \rangle^2 \,.

The form factors :math:`f(q)` of a specific atom can be approximated with

.. math::
    f(\sin\theta/\lambda) = \sum_{i=1}^4 a_i e^{-b_i \sin^2\theta/\lambda^2} + c \,.

Expressed in terms of the scattering vector we can write

.. math::
    f(q) = \sum_{i=1}^4 a_i e^{-b_i q^2/(4\pi)^2} + c \,.

The coefficients :math:`a_{1,\dots,4}`, :math:`b_{1,\dots,4}` and :math:`c` 
are documented in :footcite:t:`princeInternationalTablesCrystallography2004`.

References
----------
.. footbibliography::
