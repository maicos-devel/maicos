#!/usr/bin/env python
#
# Copyright (c) 2025 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Setup file for MAICoS package.

Credit to MDAnalysis setup.py.
"""

import os
import shutil
import sys
import tempfile
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
from pathlib import Path

import numpy as np
from setuptools import Extension, setup

import versioneer

VERSION = versioneer.get_version()
is_release = "+" not in VERSION


def hasfunction(cc, funcname, include=None, extra_postargs=None):
    """Check for function.

    Credit to MDAnalysis setup.py.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="hasfunction-"))
    devnull = oldstderr = None
    try:
        try:
            fname = tmpdir / "funcname.c"
            with Path(fname).open(mode="w") as f:
                if include is not None:
                    f.write(f"#include {include!s}\n")
                f.write("int main(void) {\n")
                f.write(f"    {funcname!s};\n")
                f.write("}\n")
            # Redirect stderr to /dev/null to hide any error messages from the compiler.
            # This will have to be changed if we ever have to check for a function on
            # Windows.
            oldstderr = os.dup(sys.stderr.fileno())
            with Path("/dev/null", "w").open() as devnull:
                os.dup2(devnull.fileno(), sys.stderr.fileno())

            objects = cc.compile(
                [fname], output_dir=str(tmpdir), extra_postargs=extra_postargs
            )
            cc.link_executable(objects, str(tmpdir / "a.out"))
        except Exception:
            return False
        return True
    finally:
        if oldstderr is not None:
            os.dup2(oldstderr, sys.stderr.fileno())
        shutil.rmtree(tmpdir)


def detect_openmp():
    """Support for OpenMP parallelization.

    Check if this compiler support OpenMP parallelization. Credit to MDAnalysis
    setup.py.
    """
    print("Attempting to autodetect OpenMP support... ", end="")  # noqa: T201
    compiler = new_compiler()
    customize_compiler(compiler)
    compiler.add_library("gomp")
    include = "<omp.h>"
    extra_postargs = ["-fopenmp"]
    hasopenmp = hasfunction(
        compiler,
        "omp_get_num_threads()",
        include=include,
        extra_postargs=extra_postargs,
    )
    if hasopenmp:
        print("Compiler supports OpenMP")  # noqa: T201
    else:
        print("Did not detect OpenMP support.")  # noqa: T201
    return hasopenmp


if __name__ == "__main__":
    # Windows automatically handles math library linking and will not build if we try to
    # specify one
    mathlib = [] if os.name == "nt" else ["m"]

    has_openmp = detect_openmp()
    use_cython = not is_release or bool(os.getenv("USE_CYTHON"))
    source_suffix = ".pyx" if use_cython else ".c"

    pre_exts = [
        Extension(
            "maicos.lib._cmath",
            ["src/maicos/lib/_cmath" + source_suffix],
            include_dirs=[np.get_include()],
            extra_compile_args=["-std=c99", "-ffast-math", "-O3", "-funroll-loops"]
            + has_openmp * ["-fopenmp"],
            extra_link_args=has_openmp * ["-fopenmp"],
            libraries=mathlib,
        )
    ]

    if use_cython:
        from Cython.Build import cythonize

        extensions = cythonize(pre_exts, force=True)
    else:
        extensions = pre_exts
        # Let's check early for missing .c files
        for ext in extensions:
            for source in ext.sources:
                if not (Path(source).exists() and os.access(source, os.R_OK)):
                    raise OSError(
                        f"Source file '{source}' not found. This might be caused by a "
                        "missing Cython install, or a failed/disabled Cython build."
                    )

    setup(cmdclass=versioneer.get_cmdclass(), version=VERSION, ext_modules=extensions)
