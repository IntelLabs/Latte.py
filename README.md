# Basic Setup

Download and install Python 3 if not available on your system.  I find
Miniconda (from Anaconda) to be the simplest, lightweight solution for
cross-platform Python management.  It also simplifies configuration by
using a local Python installation that won't conflict with other users.

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

For development, install Latte's current working directory as a "symbolic"
package.  This simplifies testing as it tells Python to load the latest changes
to any files you are working on.
```
pip install -e .
```

# Tests
```
py.test test
```
Checkout the [py.test documentation](http://pytest.org/latest/index.html) for
more detailed usage documentation.
