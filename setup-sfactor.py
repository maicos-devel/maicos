from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

os.environ["CC"] = "gcc-7"
os.chdir("sfactor")

setup(
  name = "sfactor",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("sfactor",
              ["sfactor.pyx"],
              include_dirs=[np.get_include()],
              extra_compile_args = ["-O3", "-fopenmp"],
              extra_link_args=['-fopenmp']
              )
  ]
)
