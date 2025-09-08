python_sources = [
    '__init__.py',
    'dust.py',
    'gas.py',
    'sim.py',
    'compo.py',
    'component.py',
    'addcomponent.py'
]

py3.install_sources(python_sources, subdir: 'tripod/std')

dust_fortran_sources = [
    '../constants/constants.f90',
    '../utils/interpolation.f90',
    'dust.f90',
]

dust_source = custom_target(
    'dust_fmodule.c',
    input: dust_fortran_sources,
    output: [
        'dust_fmodule.c',
        'dust_f-f2pywrappers2.f90',
    ],
    command: [
        py3,
        '-m', 'numpy.f2py',
        '--lower',
        '-m', 'dust_f',
        '@INPUT@',
        '--build-dir', 'tripod/std',
    ],
)

py3.extension_module(
    'dust_f',
    dust_fortran_sources + dust_source,
    incdir_f2py / 'fortranobject.c',
    include_directories: inc_np,
    dependencies: py_dep,
    install: true,
    subdir: 'tripod/std',
)
