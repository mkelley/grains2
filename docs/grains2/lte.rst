Grain temperatures
==================

The ``PlaneParallelIsotropicLTE`` class is used to calculate the temperature of a dust grain in local thermodynamic equilibrium between absorbed sunlight (plane parallel waves) and re-radiated thermal energy (isotropic emission).

Solid grains
------------

To calculate the temperature of a 1.0 μm solid amorphous carbon grain:

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


Porous grains
-------------

The LTE models have support for porous grains.  Because their porosity may depend
on their sizes, the LTE models have an optional parameter named ``porosity``.  See `porosity`_ for more details.  Here we repeat the above excercise, but with a fractal porosity model (sub-unit radius = 0.1 μm, fractal dimension = 2.8):

   >>> from grains2 import FractalPorosity
   >>>
   >>> porosity = FractalPorosity(a0=0.1, D=2.8)
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh, porosity=porosity)
   >>> print(lte.T, "K")  # doctest: +FLOAT_CMP
   [505.22681188 278.26006943 205.54160889] K

The temperature of the smallest grain, 0.1 μm, is unchanged because they are solid in this model.  The porosity increases with grain size, and the larger grains have temperatures different from the solid grains.

Plot the temperatures of amorphous carbon grains at 1.5 au for fractal dimensions from 2.5 to 3.0:

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from grains2 import PlaneParallelIsotropicLTE, FractalPorosity, amcarbon

   D = [2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
   a = np.logspace(-1, 2)
   ac = amcarbon()
   rh = 1.5
   T = np.zeros((6, len(a)))

   fig, ax = plt.subplots()

   for i in range(len(D)):
       porosity = FractalPorosity(0.1, D[i])
       lte = PlaneParallelIsotropicLTE(a, ac, rh, porosity=porosity)
       ax.plot(a, lte.T, label=D[i])

   ax.legend()
   plt.setp(
       ax,
       xlabel="Radius (μm)",
       xscale="log",
       xlim=(0.1, 100),
       ylabel="$T$",
   )
   plt.tight_layout()
