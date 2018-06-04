============================================================================
Cython example of exposing C-computed arrays in Python without data copies
============================================================================

The goal of this example is to show how an existing C codebase for
numerical computing (here `c_code.c`) can be wrapped in Cython to be
exposed in Python. 

The meat of the example is that the data is allocated in C, but exposed
in Python without a copy using the `PyArray_SimpleNewFromData` numpy
function in the Cython file `cython_wrapper.pyx`.

The purpose of the `ArrayWrapper` object, is to be garbage-collected by
Python when the ndarray Python object disappear. The memory is then
freed. Note that there is no control of when Python will deallocate the
memory. If the memory is still being used by the C code, please refer to
the following blog post by Travis Oliphant:

 http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory

You will need Cython, numpy, and a C compiler. 

To build the C extension in-place run::

    $ python setup.py build_ext -i


To test the C-Python bindings, run the `test.py` file.

================= =========================================================
Files
================= =========================================================
c_code.c          The C code to bind. Knows nothing about Python
cython_wrapper.c  The Cython code implementing the binding
setup.py          The configure/make/install script
test.py           Python code using the C extension
================= =========================================================

____

:Author: Gael Varoquaux
:License: BSD
