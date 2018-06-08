from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_setuptools
import subprocess
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
    print('CYTHON NOT FOUND: DISTRIBUTING VERSION (USING PROVIDED dasslc.c module source)')
else:
    USE_CYTHON = True
import numpy as np

NAME = "dasslcy"
VERSION = "0.1"
DESCR = "A cython wrapper for dasslc"
URL = "http://www.google.com"
REQUIRES = ['numpy']

AUTHOR = "Caio Marcellos"
EMAIL = "caiocuritiba@gmail.com"

LICENSE = "MIT"

SRC_DIR = "dasslcy"
PACKAGES = [SRC_DIR]

EXTRA_SOURCES = ["./dasslc_base/dasslc.c"]

def download_dasslc():
    print("RUNNING SCRIPT to retrieve dasslc source code")
    # subprocess.call(["sh", "./get_dasslc.sh"])
    from get_dasslc import get_dasslc
    get_dasslc()

## Preprocess before building extension
class BuildExtCommand(build_ext_setuptools):
    """Download dasslc source code before installing dasslcy"""
    def run(self):
        download_dasslc()
        build_ext_setuptools.run(self)

### EXTENSION
ext = '.pyx' if USE_CYTHON else '.c'
ext_1 = Extension(SRC_DIR + ".dasslc",
                  [SRC_DIR + "/dasslc{}".format(ext)] + EXTRA_SOURCES,
                  libraries=[],
                  include_dirs=[np.get_include()])

EXTENSIONS = [ext_1]

CMDCLASS = {"build_ext": BuildExtCommand}

if USE_CYTHON:
    EXTENSIONS = cythonize([ext_1])
else:
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
          cmdclass=CMDCLASS,
          ext_modules=EXTENSIONS
          )
