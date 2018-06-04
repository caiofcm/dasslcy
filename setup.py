from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension("dasslc", ["dasslc.pyx"],
                                     extra_compile_args=['-std=c11'])])
)
