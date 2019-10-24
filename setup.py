#!/usr/bin/env python3
from glob import glob
from setuptools import find_packages
from numpy.distutils.core import setup, Extension

ext = [Extension(name='grains2.bhmie.bhmie', sources=['src/bhmie/bhmie.f']),
       Extension(name='grains2.davint.davint', sources=glob('src/davint/*.f'))]

if __name__ == "__main__":
    setup(name='grains2',
          version='2.0.0.dev0',
          description='Dust and ice grain models',
          author="Michael S. P. Kelley",
          author_email="msk@astro.umd.edu",
          packages=find_packages(),
          package_data={'grains2': ['data/*']},
          ext_modules=ext,
          scripts=['scripts/water-ice-lifetime'],
          requires=['numpy', 'astropy'],
          #          license='BSD',
          )
