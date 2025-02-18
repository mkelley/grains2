.. grains2 documentation master file, created by
   sphinx-quickstart on Tue Feb 18 10:44:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

grains2 documentation
=====================

Use ``grains2`` to calculate the temperature or spectrum of dust in equilibrium with radiation, with optional considerations for sublimation.  ``grains2`` is designed for observations of solar system dust at a single location with respect to the Sun, but implementations for other scenarios is possible (e.g., an exocomet around beta Pic).

The concepts are primarily based on the Hanner-Harker dust model for comets (Harker et al. 2002, 2007).  However, support for cometary crystalline silicates is incomplete.

grains2 is primarily Mie-based.  Bohren and Huffman Mie code is from Bruce Draine https://www.astro.princeton.edu/~draine/scattering.html  See source code for improvements on the original BH code.


Refractory dust
---------------

Dust grains are described by indices of refraction, which are encapsulated within instances of the ``Material`` class:

   >>> from grains2 import amcarbon
   >>> ac = amcarbon()
   >>> ac
   <Material [amorphous carbon]>

The ``PlaneParallelIsotropicLTE`` class is used to calculate the temperature of a dust grain in local thermodynamic equilibrium between absorbed sunlight (plane parallel waves) and re-radiated thermal energy (isotropic emission).

To calculate the temperature of a 1.0 μm amorphous carbon grain:

   >>> from grains2 import PlaneParallelIsotropicLTE
   >>> a = 1.0  # radius, μm
   >>> rh = 1.5  # heliocentric distance, au
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh)
   >>> print(lte.T)  # doctest: +FLOAT_CMP
   [268.95336711]

Arrays of grain sizes are also allowed:

   >>> a = [0.1, 1, 10]  # radius, μm
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh)
   >>> print(lte.T)  # doctest: +FLOAT_CMP
   [505.22681188 268.95336711 219.08381423]


Ice sublimation
---------------

The sublimation model balances insolation with energy lost from sublimation and thermal radiation.  Losses due to solar wind sputtering (Mukai & Schwehm 1981) may also be included.  The optical constants of water ice from Warren & Brandt (2008) are included.  The latent heat of sublimation follows Delsemme & Miller (1971), and the vapor pressure equation of Lichtenegger & Komle (1991) is also used. Dust-ice aggregates may be created by mixing optical constants with the effective medium approximation of Bruggeman (1935).


.. toctree::
   :maxdepth: 2
   :caption: Contents:

