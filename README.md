# TriPoDPy

## Dust Coagulation and Evolution in Protoplanetary Disks

`TriPoDPy` is a Python package to simulate the evolution of dust in protoplanetary disks using a parametric dust evolution model.

`TriPoDPy` simulates the radial evolution of gas and dust in a protoplanetary disk, involving viscous evolution of the gas disk, advection and diffusion of the dust disk, as well as dust growth via the TriPoD method [Pfeil+ 2024](https://ui.adsabs.harvard.edu/abs/2024A&A...691A..45P).

Please read the [documentation](https://tripodpy.readthedocs.io/en/latest/) for a detailed description. The examples in the documentation can be found in the ``docs/source`` folder.

`TriPoDPy` has been submitted to the Journal of Open Source Software(JOSS) and is awaiting review. There are currently a number of planned publications using `TriPoDPy` in the works, so if you plan on using `TriPoDPy` in your work, consider contacting us to avoid conflicts/duplication of work. We are always happy to collaborate/add new features.

`TriPoDPy` is free software distributed under the terms of the GNU General Public License v3.0, see [LICENSE](https://github.com/tripod-code/tripodpy/blob/main/LICENSE).

## Requirements

- Python 3.11 or newer (tested with 3.11 – 3.13)
- [DustPy](https://github.com/stammler/dustpy) 1.0.8 or newer and [Simframe](https://github.com/stammler/simframe) 1.0.6 or a later 1.0.x release. `pip` installs both automatically from PyPI, so the GitHub versions are not needed.

- A Fortran compiler, e.g. `gfortran` (GCC 9 or newer). It is needed to compile `TriPoDPy`'s Fortran routines during installation.
- Linux or macOS (Intel and Apple Silicon). On Windows, please use [WSL](https://learn.microsoft.com/windows/wsl/install).

## Installation

First install a Fortran compiler (`sudo apt install gfortran` on Debian/Ubuntu, `brew install gcc` on macOS) and check that it can be found with `gfortran --version`. Then install `TriPoDPy` into a fresh virtual environment:

```bash
git clone https://github.com/tripod-code/tripodpy.git
cd tripodpy
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Alternatively, conda users can use the provided `environment.yml`, which also includes a Fortran compiler: `conda env create -f environment.yml`, `conda activate tripodpy`, `pip install .`.

For an editable installation, use `pip install --no-build-isolation --editable .`. In this case `meson-python`, `ninja` and `numpy` need to be installed in your environment.

See the [installation guide](https://tripodpy.readthedocs.io/en/latest/installation.html) for detailed instructions and troubleshooting.

## Contents

- `tripodpy/` – the Python package
  - `simulation.py` – the `Simulation` class, which sets up and runs a model
  - `std/` – standard physics for gas, dust (TriPoD growth, drift and diffusion; Fortran kernels in `dust.f90`), and compositional tracking (`addcomponent`, `compo`, `component`)
  - `utils/` – helpers for reading output (`read_data`) and reconstructing the particle size distribution
  - `plot/` – quick-look plotting routines (`panel`, `ipanel`)
  - `constants/` – physical constants
- `docs/` – Sphinx documentation. `docs/source` holds the tutorial notebooks.
- `unit_tests/` – `pytest` unit tests, run by the GitHub Actions workflows in `.github/workflows/`
- `tests/` – example setups (planetary gap, ice lines) and notebooks that compare `TriPoDPy` with full-coagulation `DustPy` runs
- `paper/` – the JOSS paper (`paper.md`), its bibliography and figures


## Framework

`TriPoDPy` is using the [Simframe](http://github.com/stammler/simframe/) framework for scientific simulations ([Stammler & Birnstiel 2022](https://joss.theoj.org/papers/0ef61e034c57445e846b2ec383c920a6))

## Contributing and support

Contributions, bug reports and questions are very welcome. Please see the [contributing guide](https://github.com/tripod-code/tripodpy/blob/main/.github/CONTRIBUTING.md) for details.

- **Bug reports:** open a [bug report](https://github.com/tripod-code/tripodpy/issues/new?template=bug_report.md) on the issue tracker. Please include a minimal script that reproduces the problem, the full error message, and the versions of `TriPoDPy`, `DustPy`, `Simframe` and Python.
- **Questions and support:** open a [question](https://github.com/tripod-code/tripodpy/issues/new?template=question.md) on the issue tracker or contact the maintainers (e.g. n.kaufmann@usm.lmu.de).
- **Feature requests:** open a [feature request](https://github.com/tripod-code/tripodpy/issues/new?template=feature_request.md).
- **Code contributions:** fork the repository, create a branch, and open a pull request against the `main` branch. Please run the unit tests with `pytest unit_tests` first. They also run automatically on every pull request.

Before starting any significant development, please contact the maintainers to avoid duplicating work.

## Acknowledgements

`TriPoDPy` was developed at the [University Observatory](https://www.usm.uni-muenchen.de/index_en.php) of the [Ludwig Maximilian University of Munich](https://www.en.uni-muenchen.de/index.html). The authors acknowledge funding from the European Union under the European Unionʼs Horizon Europe Research and Innovation Programme 101124282 (EARLYBIRD) and funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC-2094 - 390783311. Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
