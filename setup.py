# from setuptools import setup, find_packages
# from setuptools.extension import Extension
# from Cython.Build import cythonize

# extensions = [
#     Extension(
#         "dasslc_cy_wrapper.dasslc",
#         ["dasslc_cy_wrapper/dasslc.pyx", "dasslc/dasslc.c"],
#         # not needed for fftw unless it is installed in an unusual place
#         include_dirs=['./dasslc/'],
#         # libraries=['fftw3', 'fftw3f', 'fftw3l', 'fftw3_threads',
#         #           'fftw3f_threads', 'fftw3l_threads'],
#         #library_dirs=['/dasslc/'],  # numpy.get_include()
#         extra_compile_args=['-std=c11']
#     ),
# ]

# setup(
#     name="dasslc_cy_wrapper",
#     packages=find_packages(),
#     ext_modules=cythonize(extensions)
# )

from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

NAME = "dasslcy"
VERSION = "0.1"
DESCR = "A cython wrapper for dasslc (Argimiro Resende Secchi)"
URL = "http://www.google.com"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Caio Marcellos"
EMAIL = "caiocuritiba@gmail.com"

LICENSE = "Apache 2.0"

SRC_DIR = "cython_cy_wrapper"
PACKAGES = [SRC_DIR]

ext_1 = Extension(SRC_DIR + ".dasslc",
                  ["/dasslc/dasslc.c", SRC_DIR + "/dasslc.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(install_requires=REQUIRES,
          packages=PACKAGES,
          zip_safe=False,
          name=NAME,
          version=VERSION,
          description=DESCR,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          license=LICENSE,
          cmdclass={"build_ext": build_ext},
          ext_modules=EXTENSIONS
          )
