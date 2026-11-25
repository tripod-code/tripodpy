Contributing and support
========================

``TriPoDPy`` is developed openly on `GitHub <https://github.com/tripod-code/tripodpy>`_, and contributions,
bug reports and questions are very welcome. The full guidelines are in the
`contributing guide <https://github.com/tripod-code/tripodpy/blob/main/.github/CONTRIBUTING.md>`_.

Several projects using ``TriPoDPy`` are currently in progress. Before you start a larger development, please
contact the maintainers (e.g. n.kaufmann@usm.lmu.de) so that we can avoid duplicated work.

Reporting bugs
--------------

If you find a bug or unexpected behaviour, please open a
`bug report <https://github.com/tripod-code/tripodpy/issues/new?template=bug_report.md>`_ on the issue tracker.
A good bug report contains

* a short description of the problem and of the behaviour you expected,
* a minimal script that reproduces the problem,
* the full error message, if there is one,
* the versions of ``TriPoDPy``, ``DustPy``, ``Simframe`` and Python, your operating system and your Fortran compiler.

You can get the versions with:

.. code-block:: bash

   pip list | grep -i -E "tripodpy|dustpy|simframe"
   python --version
   gfortran --version

Asking questions and getting support
------------------------------------

If your question is not answered in this documentation, please open a
`question <https://github.com/tripod-code/tripodpy/issues/new?template=question.md>`_ on the issue tracker or
contact the maintainers by email.

Requesting features
-------------------

If you are missing a feature or have an idea for an improvement, please open a
`feature request <https://github.com/tripod-code/tripodpy/issues/new?template=feature_request.md>`_.

Contributing code
-----------------

New contributions are proposed through GitHub pull requests:

1. Fork ``TriPoDPy`` on GitHub, clone your fork and create a new branch for your changes.
2. Install ``TriPoDPy`` in editable mode as described in :doc:`installation`.
3. Make your changes. Please add unit tests in ``unit_tests/`` for new functionality and document new
   features in the docstrings and, where useful, in the tutorial notebooks.
4. Run the unit tests with ``pytest unit_tests`` and make sure they pass.
5. Push your changes to your fork and open a pull request against the ``main`` branch.

The unit tests run automatically on every pull request.
