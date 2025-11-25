# TriPoDPy

## Dust Coagulation and Evolution in Protoplanetary Disks

`TriPoDPy` is a Python package to simulate the evolution of dust in protoplanetary disks using a parametric dust evolution model.

`TriPoDPy` simulates the radial evolution of gas and dust in a protoplanetary disk, involving viscous evolution of the gas disk, advection and diffusion of the dust disk, as well as dust growth via the TriPoD method [Pfeil+ 2024](https://ui.adsabs.harvard.edu/abs/2024A&A...691A..45P).

Please read the [documentation](https://tripodpy.readthedocs.io/en/latest/) for a detailed description.

`TriPoDPy` has been submitted to the Journal of Open Source Software(JOSS) and is awaiting review. There are currently a number of planned publications using `TriPoDPy` in the works, so if you plan on using `TriPoDPy` in your work, consider contacting us to avoid conflicts/duplication of work. We are always happy to collaborate/add new features.

By using any version of `TriPoDPy` you agree to these terms of usage.

## Installation

Clone the repository, then it can be installed via

`pip install .`

or

`pip install --no-build-isolation --editable .`

for editable installation. In the latter case you need to have `meson-python` and `ninja` installed.

`pip install meson-python ninja`

__Note that currently the instalation with the GitHub repository versions of [`Dustpy`](https://github.com/stammler/dustpy) and [`Simframe`](https://github.com/stammler/simframe) is recomended to ensure full compatibility__.

## Requirements

`TriPoDPy` needs a Python3 distribution and a Fortran compiler installed on your system.

## Framework

`TriPoDPy` is using the [Simframe](http://github.com/stammler/simframe/) framework for scientific simulations ([Stammler & Birnstiel 2022](https://joss.theoj.org/papers/0ef61e034c57445e846b2ec383c920a6))

## Acknowledgements

`TriPoDPy` was developed at the [University Observatory](https://www.usm.uni-muenchen.de/index_en.php) of the [Ludwig Maximilian University of Munich](https://www.en.uni-muenchen.de/index.html). The authors acknowledge funding from the European Union under the European Unionʼs Horizon Europe Research and Innovation Programme 101124282 (EARLYBIRD) and funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC-2094 - 390783311. Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
