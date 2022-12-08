from setuptools import setup, Extension, find_packages

octopy = Extension(
    '_octopy',
    sources=[
        'src/math/tensor.c',
        'src/python/py_tensor.c',
        'src/octopy_helper.c',
        'src/python/py_octopy.c',
    ]
)

setup(
    name='octopy',
    author='Kyle R. Chickering',
    author_email='kyrochickering@gmail.com',
    version='0.1.2',
    ext_modules=[octopy],
    packages=['octopy_core'],
)
