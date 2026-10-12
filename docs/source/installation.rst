Installation
============

``TriPoDPy`` contains Fortran routines that are compiled when the package is installed.
You therefore need a **Fortran compiler** in addition to Python.

Requirements
------------

* **Python** 3.11 or newer (tested with 3.11 – 3.13)
* **DustPy** 1.0.8 or newer (installed automatically by ``pip``)
* **Fortran compiler**: ``gfortran`` (GCC) 9 or newer is recommended and is the compiler used in our tests.
  Other compilers supported by `Meson <https://mesonbuild.com/>`_ (e.g. ``flang``, ``ifx``) may work but are not tested.
* **Operating systems**: Linux and macOS (Intel and Apple Silicon). On Windows, please use the
  `Windows Subsystem for Linux (WSL) <https://learn.microsoft.com/windows/wsl/install>`_.

We strongly recommend installing ``TriPoDPy`` into a fresh, dedicated environment.

Option 1: system compiler and virtual environment (recommended)
---------------------------------------------------------------

First install a Fortran compiler for your operating system:

.. list-table::
   :header-rows: 1

   * - System
     - Command
   * - Debian / Ubuntu
     - ``sudo apt install gfortran``
   * - Fedora / RHEL
     - ``sudo dnf install gcc-gfortran``
   * - macOS (`Homebrew <https://brew.sh>`_)
     - ``xcode-select --install`` and ``brew install gcc``

Make sure the compiler can be found (see `Checking the Fortran compiler`_ below), then create a virtual
environment and install ``TriPoDPy``:

.. code-block:: bash

   git clone https://github.com/tripod-code/tripodpy.git
   cd tripodpy
   python3 -m venv .venv
   source .venv/bin/activate
   pip install .

Option 2: conda environment
---------------------------

If you use conda, or cannot install a system compiler, the repository ships an ``environment.yml`` which installs
Python, a Fortran compiler and all build dependencies from `conda-forge <https://conda-forge.org/>`_.
This works on Linux and macOS without installing a system compiler.

.. code-block:: bash

   git clone https://github.com/tripod-code/tripodpy.git
   cd tripodpy
   conda env create -f environment.yml
   conda activate tripodpy
   pip install .

Checking the Fortran compiler
-----------------------------

Before installing, make sure the compiler can be found on your ``PATH``:

.. code-block:: bash

   gfortran --version
   which gfortran

If the compiler is installed under a different name (e.g. Homebrew may provide ``gfortran-14``), tell the build
which one to use via the ``FC`` environment variable:

.. code-block:: bash

   FC=gfortran-14 pip install .

Editable installation for developers
------------------------------------

For development, install ``TriPoDPy`` in editable mode. The build dependencies have to be present in the environment
(they are already included in ``environment.yml``):

.. code-block:: bash

   pip install meson-python ninja numpy
   pip install --no-build-isolation --editable .

Checking the installation
-------------------------

.. code-block:: bash

   python -c "import tripodpy; print(tripodpy.__version__)"

To run the unit tests:

.. code-block:: bash

   pip install pytest
   pytest unit_tests

Troubleshooting
---------------

**No Fortran compiler found.** An error like

.. code-block:: text

   ERROR: Unknown compiler(s): [['gfortran'], ['flang-new'], ['flang'], ...]

or ``No Fortran compiler found`` means that no Fortran compiler was found on your ``PATH``.
Install one as described above (or use the conda environment) and check it with ``gfortran --version``.

**Dependency versions.** ``TriPoDPy`` builds on `DustPy <https://github.com/stammler/dustpy>`_ and
`Simframe <https://github.com/stammler/simframe>`_. If you run into incompatibilities, install their latest
versions from GitHub:

.. code-block:: bash

   pip install git+https://github.com/stammler/simframe.git git+https://github.com/stammler/dustpy.git
