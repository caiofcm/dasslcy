from setuptools import setup, Extension
import setuptools.command.build_py
import subprocess
from Cython.Distutils import build_ext
import numpy as np

NAME = "dasslcy"
VERSION = "0.1"
DESCR = "A cython wrapper for dasslc (Argimiro Resende Secchi)"
URL = "http://www.google.com"
REQUIRES = ['numpy', 'cython']

AUTHOR = "Caio Marcellos"
EMAIL = "caiocuritiba@gmail.com"

LICENSE = "MIT"

SRC_DIR = "dasslcy"
PACKAGES = [SRC_DIR]

## PRE-INSTALLATION: DOWNLOAD DASSLC
class BuildPyCommand(setuptools.command.build_py.build_py):
    """Download dasslc source code before installing dasslcy"""

    def run(self):
        print("RUNNING SCRIPT to retrieve dasslc source code")
        subprocess.call(["sh", "./get_dasslc.sh"])
        setuptools.command.build_py.build_py.run(self)

### EXTENSION
ext_1 = Extension(SRC_DIR + ".dasslc",
                  ["./dasslc_base/dasslc.c", SRC_DIR + "/dasslc.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

CMDCLASS = {"build_ext": build_ext,
            "build_py": BuildPyCommand,
            }#"develop": CustomDevelopCommand}

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
