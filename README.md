# TriPoDPy

## Dust Coagulation and Evolution in Protoplanetary Disks

`TriPoDPy` is a Python package to simulate the evolution of dust in protoplanetary disks using a parametric dust evolution model.

`TriPoDPy` simulates the radial evolution of gas and dust in a protoplanetary disk, involving viscous evolution of the gas disk, advection and diffusion of the dust disk, as well as dust growth via the TriPoD method [Pfeil+ 2024](https://ui.adsabs.harvard.edu/abs/2024A&A...691A..45P).

Please read the [documentation](https://tripodpy.readthedocs.io/en/latest/) for a detailed description.

`TriPoDPy` has been submitted to the Journal of Open Source Software(JOSS) and is awaiting review. There are currently a number of planned publications using `TriPoDPy` in the works, so if you plan on using `TriPoDPy` in your work, consider contacting us to avoid conflicts/duplication of work. We are always happy to collaborate/add new features.

By using any version of `TriPoDPy` you agree to these terms of usage.

## Requirements

- Python 3.10 or newer (tested with 3.10 – 3.13)
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

__Note that currently the installation with the GitHub repository versions of [`Dustpy`](https://github.com/stammler/dustpy) and [`Simframe`](https://github.com/stammler/simframe) is recommended to ensure full compatibility__.

## Framework

`TriPoDPy` is using the [Simframe](http://github.com/stammler/simframe/) framework for scientific simulations ([Stammler & Birnstiel 2022](https://joss.theoj.org/papers/0ef61e034c57445e846b2ec383c920a6))

## Acknowledgements

`TriPoDPy` was developed at the [University Observatory](https://www.usm.uni-muenchen.de/index_en.php) of the [Ludwig Maximilian University of Munich](https://www.en.uni-muenchen.de/index.html). The authors acknowledge funding from the European Union under the European Unionʼs Horizon Europe Research and Innovation Programme 101124282 (EARLYBIRD) and funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC-2094 - 390783311. Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
