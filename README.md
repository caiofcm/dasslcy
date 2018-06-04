# Python3 wrapper for DASSLC based on Cython

This is a simple wrapper for using the "Differential-Algebraic System Solver in C" by Argimiro R. Secchi (PEQ/COPPE/UFRJ) in python3 using Cython.

## Getting Started

All the information about the usage of this package can be found in example file
```
Dasslc2py/dasslc_examples.py
```

### Prerequisites

- Mingw for Windows with Python3 and C compiler (see [python-mingw]):


### Installing

In order to locally build this module, open a terminal at the dir **Dasslc2py** and run the following command

```
python2 setup.py build_ext --inplace
```

Instead, if you want to install it along with your python distribution, run (as root)
```
python2 setup.py install
```

## Author

* **Ataíde Neto** - ataide@peq.coppe.ufrj.br

[python-mingw]: https://stackoverflow.com/questions/41932407/which-python-should-i-install-and-how-when-using-msys2

