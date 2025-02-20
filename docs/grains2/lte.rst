Grain temperatures
==================

The ``PlaneParallelIsotropicLTE`` class is used to calculate the temperature of a dust grain in local thermodynamic equilibrium between absorbed sunlight (plane parallel waves) and re-radiated thermal energy (isotropic emission).

To calculate the temperature of a 1.0 μm amorphous carbon grain:

   >>> from grains2 import PlaneParallelIsotropicLTE, amcarbon
   >>>
   >>> a = 1.0  # radius, μm
   >>> ac = amcarbon()
   >>> rh = 1.5  # heliocentric distance, au
   >>>
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh)
   >>> print(lte.T[0], "K")  # doctest: +FLOAT_CMP
   268.95336711 K

Arrays of grain sizes are also allowed:

   >>> a = [0.1, 1, 10]  # radius, μm
   >>>
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh)
   >>> print(lte.T, "K")  # doctest: +FLOAT_CMP
   [505.22681188 268.95336711 219.08381423] K
