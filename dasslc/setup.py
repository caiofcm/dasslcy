#!/usr/bin/env python2

from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
	name = 'Dasslc2py',
	version = '0.1',
	license = 'Free',
	author = 'ataide@peq.coppe.ufrj.br',
    ext_modules=[
        Extension("dasslc",
            sources = ["dasslcmodule.c","dasslc.c"],
            include_dirs=get_numpy_include_dirs(),
        )
    ]
)

# Debug flags: CFLAGS='-Wall -O0 -g'