Grain composition and refractive indices
========================================


Pure materials
--------------

Dust (and ice) grains are described by indices of refraction, which are encapsulated within instances of the ``Material`` class.  ``Material`` may be used to define the grain composition directly.  There are a set of materials provided in the ``grains2.material`` sub-module:

   >>> from grains2 import amcarbon
   >>> ac = amcarbon()
   >>> ac
   <Material [amorphous carbon]>
   >>> print("Bulk density: ", ac.rho, "g/cm3")
   Bulk density: 1.5 g/cm3


Effective medium approximation
------------------------------

The effective medium approximation is a technique used to produce indices of refraction for a mixure of two materials and is commonly used in astronomy (Kolokolova et al. 2024, Comets III).  ``grains2.ema`` includes the Bruggeman mixing rule.  Create a mixutre of amorphous olivine (Mg/Fe=50/50) and amorphous carbon with a ratio of 1/3:

   >>> from grains2 import Bruggeman, amolivine50
   >>>
   >>> ao50 = amolivine50()
   >>> mix = Bruggeman.mix([ac, ao50], [1, 3])
   >>> print("Bulk density: ", mix.rho, "g/cm3")  # doctest: +FLOAT_CMP
   Bulk density: 2.85 g/cm3

Plot the real part of the refractive indices:

.. plot::
   :context:

   import matplotlib.pyplot as plt
   from grains2 import amcarbon, amolivine50, Bruggeman

   ac = amcarbon()
   ao50 = amolivine50()
   mix = Bruggeman.mix([ac, ao50], [1, 3])

   fig, ax = plt.subplots()
   ax.plot(ac.ri.wave, ac.ri.n, label="Amorphous carbon")
   ax.plot(ao50.ri.wave, ao50.ri.n, label="Amorphous olivine (Mg/Fe=50/50)")
   ax.plot(mix.ri.wave, mix.ri.n, label="Mix")
   ax.legend()
   plt.setp(ax, xlabel="Wavelength (μm)", ylabel="$n$")
