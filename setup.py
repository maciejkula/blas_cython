import os
import platform

from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
except:
    print ('You must have Cython installed. '
           'Run sudo pip install cython to do so.')
    raise


# Use gcc for openMP on OSX
if 'darwin' in platform.platform().lower():
    os.environ["CC"] = "gcc-4.9"
    os.environ["CXX"] = "g++-4.9"


# Declare extension
extensions = [Extension("cfunctions", ["cfunctions.pyx"],
              extra_link_args=["-lblas", "-include /usr/include/cblas.h"],
              extra_compile_args=['-lblas'])]


setup(
    name='blastest',
    version='0.0.1',
    ext_modules=cythonize(extensions)
)
