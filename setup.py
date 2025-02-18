from glob import glob
from numpy.distutils.core import setup, Extension

ext = [
    Extension(name="grains2.bhmie.bhmie", sources=["src/bhmie/bhmie.f"]),
    Extension(name="grains2.bhcoat.bhcoat", sources=["src/bhcoat/bhcoat.f"]),
    Extension(name="grains2.davint.davint", sources=glob("src/davint/*.f")),
]

setup(ext_modules=ext)
