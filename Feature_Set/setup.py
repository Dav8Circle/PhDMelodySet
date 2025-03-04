from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="feature_set",
    packages=find_packages(),
    ext_modules=cythonize("Feature_Set/features.py"),
)
