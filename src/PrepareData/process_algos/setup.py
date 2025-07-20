from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="algos",
    ext_modules=cythonize("algos.pyx", language_level=3),
    include_dirs=[numpy.get_include()],
)
