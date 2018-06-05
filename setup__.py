from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "dasslc-cython-wrapper",
        ["dasslc_cython/dasslc.pyx", "dasslc/dasslc.c"],
        # not needed for fftw unless it is installed in an unusual place
        include_dirs=['/dasslc/'],
        # libraries=['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads',
        #           'fftw3f_threads', 'fftw3l_threads'],
        library_dirs=['/dasslc/'],  # numpy.get_include()
    ),
]

setup(
    name="dasslc_cython",
    packages=find_packages(),
    ext_modules=cythonize(extensions)
)
