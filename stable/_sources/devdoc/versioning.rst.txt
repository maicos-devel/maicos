Version information
===================

The version information in ``maicos.__version__`` indicates the release of MAICoS using
`semantic versioning`_.

In brief:

Given a version number MAJOR.MINOR.PATCH, we increment the

1. **MAJOR** version when we make **incompatible API changes**,
2. **MINOR** version when we **add functionality** in a **backwards-compatible** manner,
   and
3. **PATCH** version when we make backwards-compatible **bug fixes**.

However, as long as the **MAJOR** number is **0** (i.e. the API has not stabilized),
even **MINOR** increases *may* introduce incompatible API changes. As soon as we have a
1.0.0 release, the public API can only be changed in a backward-incompatible manner with
an increase in MAJOR version.

Additional labels for pre-release and build metadata are available as extensions to the
MAJOR.MINOR.PATCH format, following :pep:`440`.


.. Note:: Development versions and pre-releases have a suffix after
        the release number, such as ``0.7.0+12.gFEED2BEEF``. If you have
        problems, try out a full release (e.g. ``0.7.0``) first.


.. _`semantic versioning`: http://semver.org/


Data
----

.. autodata:: maicos.__version__
