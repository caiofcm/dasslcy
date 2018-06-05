from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([Extension("dasslc", 
        ["dasslc.pyx", "dasslc/dasslc.c"],
        include_dirs=['/dasslc/'],
        extra_compile_args=['-std=c11'])])
)
