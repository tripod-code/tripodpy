from setuptools import find_packages
from setuptools import setup
import pathlib

package_name = "twopoppy"
here = pathlib.Path(__file__).absolute().parent


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

    classifiers=[],

    packages=find_packages(),
    install_requires=["simframe"],
    include_package_data=True,
    zip_safe=False,
)
