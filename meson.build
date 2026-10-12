project('tripodpy', 'c')
if not add_languages('fortran', required: false)
    error(
        'No Fortran compiler found. TriPoDPy needs a Fortran compiler (e.g. gfortran) to be installed. ' +
        'See https://tripodpy.readthedocs.io/en/latest/installation.html',
    )
endif

py_mod = import('python')
py3 = py_mod.find_installation(pure: false)
py_dep = py3.dependency()

incdir_numpy = run_command(
    py3,
    [
        '-c',
        'import os; os.chdir(".."); import numpy; print(numpy.get_include())',
    ],
    check: true,
).stdout().strip()

incdir_f2py = run_command(
    py3,
    [
        '-c',
        'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())',
    ],
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

subdir('tripodpy')
