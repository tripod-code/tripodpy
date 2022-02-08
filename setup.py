import setuptools
import sys

try:
    from numpy.distutils.core import Extension
    from numpy.distutils.core import setup
except ImportError as exc:  # We do not have our build deps installed
    msg = "Error: {} must be installed before running the build.".format(
        exc.name)
    msg = "Please install NumPy first. You can do this with `pip install numpy`"
    print(msg)
    sys.exit(1)

import pathlib


def setup_package():

    package_name = "twopoppy"
    here = pathlib.Path(__file__).absolute().parent

    # Fortran modules
    ext_const = Extension(name="twopoppy.constants._constants_f",
                          sources=["twopoppy/constants/constants.f90"])
    ext_dust = Extension(name="twopoppy.std.dust_f",
                         sources=["twopoppy/constants/constants.f90",
                                  "twopoppy/std/dust.f90"])
    extensions = [ext_const, ext_dust]

    def read_version():
        with (here / package_name / '__init__.py').open() as fid:
            for line in fid:
                if line.startswith('__version__'):
                    delim = '"' if '"' in line else "'"
                    return line.split(delim)[1]
            else:
                raise RuntimeError("Unable to find version string.")

    setup(
        name=package_name,

        description="",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        keywords="",

        url="https://github.com/birnstiel/twopoppy3",
        project_urls={"Source Code": "https://github.com/birnstiel/twopoppy3",
                      "Documentation": ""
                      },

        author="",
        author_email="",
        maintainer="",

        version=read_version(),
        license="",

        ext_modules=extensions,

        classifiers=[],

        packages=setuptools.find_packages(),
        install_requires=["dustpy", "numpy", "simframe"],
        include_package_data=True,
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
