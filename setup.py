from distutils.core import setup, Extension

octopy = Extension(
    'octopy',
    sources=[
        'src/math/tensor.c',
        'src/python/py_tensor.c',
        'src/octopy_helper.c',
        'src/python/py_octopy.c',
    ]
)

setup(
    name='octopy',
    version='0.1',
    ext_modules=[octopy],
    py_modules=['octopy/tensor']
)
