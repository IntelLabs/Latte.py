# Latte

# Setup

## Python
Latte depends on Python 3.  The preferred method of installation
is using Miniconda, which provides a clean solution to manage
Python distributions and dependencies.  Otherwise, follow
the instructions for your distribution to install Python 3.

### Miniconda
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

# Developer Notes
The following guides are for those working on Latte internals.

## Symbolic Package Setup
For developing Latte, it is convenient to tell Python to install the
current working directoty as a "symbolic" package.  This is done with:
```bash
cd ~/path/to/repo
pip install -r requirements.txt  # Install dependencies
pip install -e .
```
Now changes will be automatically imported without having to reinstall
the package.

## Configuring target compiler
Configure the target compiler using the file `~/.ctree.cfg`, for example:
```
[c]
CC = icpc
CFLAGS = -fPIC -O3 -std=c++11 -ansi-alias -march=core-avx2 
LDFLAGS = -tbb
```

# Parallelization Strategies
Latte's parallelization strategy is configured with the environment variable
`LATTE_PARALLEL_STRATEGY`.  Supported values are:  

* `SIMPLE_LOOP`        - nested parallel_for (like basic TBB)
* `FLOWGRAPH_LOOP`     - FlowGraph model for first level parallelism, parallel_for for nested
* `OPENMP`             - pragma omp parallel for (supports collapse(2))
* `OPENCL_SIMPLE_LOOP` - converts parallel loops to NDRange kernel

**Note: ** When using the OpenCL or OpenMP based parallelization strategies, be
sure to update your `~/.ctree.cfg` to contain the correct flags so the latte
compiler includes and links the proper files.  For example, OpenMP strategies
need the `-openmp` compiler flag.

# Testing
Latte **py.test** package for testing purposes.  The entire test suite
can be run with with:
```bash
py.test test
```
or individual tests with
```bash
py.test test/<test_file>.py
```
Checkout the [py.test documentation](http://pytest.org/latest/index.html) for
more detailed usage documentation.
