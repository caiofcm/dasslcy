# dasslcy: Python3 wrapper for DASSLC using Cython

This is a simple wrapper for using the "Differential-Algebraic System Solver in C" by Argimiro R. Secchi (PEQ/COPPE/UFRJ) in python3 using Cython.

And an alternative wrapper based on the Python/C-API wrapper from [dasslc2py](https://www.enq.ufrgs.br/enqlib/numeric/)

## Getting Started

All the information about the usage of this package can be found in example file

```
./tests/dasslc_example.py
```

### Prerequisites

- For windows: msys2 with Python3 and C compiler (see [python-mingw]):
- For linux: python3

### Installing

- Close this repository `git clone https://github.com/caiofcm/dasslcy.git`
- `cd dasslcy`
- Install:
    - `pip install -r requirements.txt .` for system installation
    - `pip install -r requirements.txt -e .` for development
- `pytest ./tests` or `python ./tests/dasslc_example.py`



## Authors

- Caio Marcellos
- Author of Python/C-Api Wrapper: Ataíde Neto - ataide@peq.coppe.ufrj.br

[python-mingw]: https://stackoverflow.com/questions/41932407/which-python-should-i-install-and-how-when-using-msys2

