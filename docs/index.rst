grains2 documentation
=====================

Use ``grains2`` to calculate the temperature and spectrum of dust in equilibrium with radiation, with optional considerations for sublimation.  ``grains2`` is designed for observations of solar system dust at a single location with respect to the Sun, but implementations for other scenarios is possible (e.g., an exocomet around beta Pic).

The concepts are primarily based on the Hanner-Harker dust model for comets (Harker et al. 2002, 2007).  However, support for cometary crystalline silicates is incomplete.

grains2 is primarily Mie-based.  Bohren and Huffman Mie code is from `Bruce Draine <https://www.astro.princeton.edu/~draine/scattering.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   grains2/materials
   grains2/lte
   grains2/spectra
   grains2/gsd
   grains2/sublimation
