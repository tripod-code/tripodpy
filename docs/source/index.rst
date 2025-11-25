Welcome to TripodPy's documentation!
====================================

| ``TripodPy`` is a Python package to simulate the evolution of dust in protoplanetary disks.

| This documentation is for ``TripodPy v1.0.0``.

| ``TripodPy`` simulates the radial evolution of gas and dust in protoplanetary disks, including viscous evolution of the gas, advection and diffusion of the dust, as well as dust growth by an the TriPoD Method.

| The ``TriPoD`` method has been published in `Pfeil et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024A&A...691A..45P>`_.
| ``TriPoDPy`` has been submitted to the Journal of Open Source Software(JOSS) and is awaiting review. There are currently a number of planned publications using ``TriPoDPy`` in the works, so if you plan on using ``TriPoDPy`` in your work, consider contacting us to avoid conflicts/duplication of work. We are always happy to collaborate/add new features.

| ``TripodPy`` requires a Python3 distribution and a Fortran compiler.

| ``TripodPy`` can be installed via pip after cloning the repository:

.. code-block:: bash

   pip install .

| ``TripodPy`` is based on the ``Simframe`` framework for scientific simulation (`Stammler & Birnstiel 2022 <https://joss.theoj.org/papers/10.21105/joss.03882>`_).
| Please have a look at the `Simframe Documentation <https://simframe.rtfd.io/>`_ for details of its usage.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   0_algorithm
   1_basics
   2_simple_customization
   3_advanced_customization
   4_standard_model
   5_compositional_tracking
   Example_disk_composition
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
