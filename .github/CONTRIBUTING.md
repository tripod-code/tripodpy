# Contributing to TriPoDPy

Contributions to `TriPoDPy` are very welcome. You can help by reporting bugs, asking questions, suggesting new features, or contributing code and documentation.

Several projects using `TriPoDPy` are currently in progress. Before you start a larger development, please contact the maintainers (e.g. n.kaufmann@usm.lmu.de) so that we can avoid duplicated work.

## Reporting bugs

If you find a bug or unexpected behaviour, please open a [bug report](https://github.com/tripod-code/tripodpy/issues/new?template=bug_report.md) on the issue tracker. A good bug report contains:

- a short description of the problem and of the behaviour you expected,
- a minimal script that reproduces the problem,
- the full error message, if there is one,
- the versions of `TriPoDPy`, `DustPy`, `Simframe` and Python, your operating system and your Fortran compiler.

You can get the versions with:

```bash
pip list | grep -i -E "tripodpy|dustpy|simframe"
python --version
gfortran --version
```

Please make sure that the problem can be reproduced with the code in this repository. We cannot guarantee that `TriPoDPy` works out of the box with your own custom extensions.

## Asking questions and getting support

Please first have a look at the [documentation](https://tripodpy.readthedocs.io/en/latest/), which contains several tutorial notebooks. If your question is not answered there, open a [question](https://github.com/tripod-code/tripodpy/issues/new?template=question.md) on the issue tracker or contact the maintainers by email.

## Requesting features

If you are missing a feature or have an idea for an improvement, please open a [feature request](https://github.com/tripod-code/tripodpy/issues/new?template=feature_request.md).

## Contributing code

New contributions are proposed through GitHub pull requests:

1. Fork `TriPoDPy` by clicking *Fork* in the top-right corner of the GitHub page.
2. Clone your fork and create a new branch for your changes, e.g. `git switch -c my-contribution`.
3. Install `TriPoDPy` in editable mode:
   ```bash
   pip install meson-python ninja numpy
   pip install --no-build-isolation --editable .
   ```
4. Make your changes. Please
   - add unit tests in `unit_tests/` for new functionality,
   - document new features in the docstrings and, where useful, in the notebooks in `docs/source/`.
5. Run the unit tests and make sure they pass:
   ```bash
   pip install pytest
   pytest unit_tests
   ```
6. Commit and push your changes to your fork and open a pull request against the `main` branch. Please describe what your changes do and link related issues.

The unit tests run automatically on every pull request.

## License

`TriPoDPy` is distributed under the GNU General Public License v3.0. Contributions are distributed under the same license.
