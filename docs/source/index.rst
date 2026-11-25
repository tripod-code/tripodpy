Welcome to TripoDPy's documentation!
====================================

| ``TripoDPy`` is a Python package to simulate the evolution of dust in protoplanetary disks.

| This documentation is for ``TripoDPy v1.0.0``. All the following examples are IPython notebooks and can be found in the ``docs/source`` folder of the ``TripoDPy`` repository.

| ``TripoDPy`` simulates the radial evolution of gas and dust in protoplanetary disks, including viscous evolution of the gas, advection and diffusion of the dust, as well as dust growth by the TriPoD method.

| The ``TriPoD`` method has been published in `Pfeil et al. 2024 <https://ui.adsabs.harvard.edu/abs/2024A&A...691A..45P>`_.
| ``TriPoDPy`` has been submitted to the Journal of Open Source Software(JOSS) and is awaiting review. There are currently a number of planned publications using ``TriPoDPy`` in the works, so if you plan on using ``TriPoDPy`` in your work, consider contacting us to avoid conflicts/duplication of work. We are always happy to collaborate/add new features.

| ``TripoDPy`` requires Python 3.11 or newer and a Fortran compiler (e.g. ``gfortran``).
| Please see the :doc:`installation` page for detailed instructions.

| ``TripoDPy`` is based on the ``Simframe`` framework for scientific simulation (`Stammler & Birnstiel 2022 <https://joss.theoj.org/papers/10.21105/joss.03882>`_).
| Please have a look at the `Simframe Documentation <https://simframe.rtfd.io/>`_ for details of its usage.

Statement of Need
-----------------

Simulating the dust evolution in protoplanetary disks, including growth and transport, is vital to
understanding planet formation and the structure of protoplanetary disks.

* ``TriPoDPy`` describes the vertically integrated dust size distribution with a truncated power law.
  This gives the simulation full access to the dust size distribution, which is essential to accurately
  model the dust evolution and additional physical effects like photoevaporation.
* ``TriPoDPy`` enables the addition of tracers in gas and dust, which can be used to track chemical
  composition, electrical charge, and other parameters.

All these features make ``TriPoDPy`` a valuable tool for astronomers to quickly model the dust evolution
of protoplanetary disks.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   Quick_start
   0_algorithm
   1_basics
   2_simple_customization
   3_advanced_customization
   4_standard_model
   5_compositional_tracking
   Example_disk_composition
   api
   contributing



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
