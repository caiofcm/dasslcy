from setuptools import setup, Extension
import setuptools.command.build_py
from setuptools.command.install import install
# from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
import subprocess
#from Cython.Distutils import build_ext
try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True


import numpy as np

# FILE REFERENCED FROM: https://github.com/thearn/simple-cython-example

#TODO: installation without cython for distribution (when the c file is packed)

NAME = "dasslcy"
VERSION = "0.1"
DESCR = "A cython wrapper for dasslc"
URL = "http://www.google.com"
#REQUIRES = ['numpy', 'cython']
REQUIRES = ['numpy']

AUTHOR = "Caio Marcellos"
EMAIL = "caiocuritiba@gmail.com"

LICENSE = "MIT"

SRC_DIR = "dasslcy"
PACKAGES = [SRC_DIR]

EXTRA_SOURCES = ["./dasslc_base/dasslc.c"]

if USE_CYTHON:
    class BuildExtCommand(build_ext):
        """Download dasslc source code before installing dasslcy"""

        def run(self):
            print("RUNNING SCRIPT to retrieve dasslc source code")
            subprocess.call(["sh", "./get_dasslc.sh"])
            build_ext.run(self)
# else:
#     BuildExtCommand = build_ext


### EXTENSION
ext = '.pyx' if USE_CYTHON else '.c'

ext_1 = Extension(SRC_DIR + ".dasslc",
                  [SRC_DIR + "/dasslc{}".format(ext)] + EXTRA_SOURCES,
                  libraries=[],
                  include_dirs=[np.get_include()])

EXTENSIONS = [ext_1]

if USE_CYTHON:
    CMDCLASS = {"build_ext": BuildExtCommand}
else:
    CMDCLASS = {}

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
