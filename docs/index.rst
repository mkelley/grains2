grains2 documentation
=====================

Use ``grains2`` to calculate the temperature and spectrum of dust in equilibrium with radiation, with optional considerations for sublimation.  ``grains2`` is designed for observations of solar system dust at a single location with respect to the Sun, but implementations for other scenarios is possible (e.g., an exocomet around beta Pic).

The concepts are primarily based on the Hanner-Harker dust model for comets (Harker et al. 2002, 2007).  However, support for cometary crystalline silicates is incomplete.

grains2 is primarily Mie-based.  Bohren and Huffman Mie code is from `Bruce Draine <https://www.astro.princeton.edu/~draine/scattering.html>`_.

.. contents:: Table of contents

Grain composition and refractive indices
----------------------------------------

Pure materials
^^^^^^^^^^^^^^

Dust (and ice) grains are described by indices of refraction, which are encapsulated within instances of the ``Material`` class.  ``Material`` may be used to define the grain composition directly.  There are a set of materials provided in the ``grains2.material`` sub-module:

   >>> from grains2 import amcarbon
   >>> ac = amcarbon()
   >>> ac
   <Material [amorphous carbon]>
   >>> print("Bulk density: ", ac.rho, "g/cm3")
   Bulk density: 1.5 g/cm3


Effective medium approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Grain temperatures
------------------

The ``PlaneParallelIsotropicLTE`` class is used to calculate the temperature of a dust grain in local thermodynamic equilibrium between absorbed sunlight (plane parallel waves) and re-radiated thermal energy (isotropic emission).

To calculate the temperature of a 1.0 μm amorphous carbon grain:

   >>> from grains2 import PlaneParallelIsotropicLTE
   >>> a = 1.0  # radius, μm
   >>> rh = 1.5  # heliocentric distance, au
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh)
   >>> print(lte.T[0], "K")  # doctest: +FLOAT_CMP
   268.95336711 K

Arrays of grain sizes are also allowed:

   >>> a = [0.1, 1, 10]  # radius, μm
   >>> lte = PlaneParallelIsotropicLTE(a, ac, rh)
   >>> print(lte.T, "K")  # doctest: +FLOAT_CMP
   [505.22681188 268.95336711 219.08381423] K


Ice sublimation
---------------

Ice sublimation is modeled with the ``SublimationLTE`` class, which balances absorbed sunlight with the energy losses from re-radiated thermal energy and the latent heat of sublimation of the ice.

The ``grains2`` water ice material includes the optical constants of water ice from Warren & Brandt (2008).  The latent heat of sublimation follows Delsemme & Miller (1971), and the vapor pressure equation of Lichtenegger & Komle (1991) is also used.


Temperature and mass-loss rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the temperature and sublimation rate of a 1.0 μm pure water ice grain at 1.0 au from the Sun:

   >>> from grains2 import SublimationLTE, waterice
   >>>
   >>> a = 1.0  # radius, μm
   >>> ice = waterice()
   >>> rh = 1.0  # heliocentric distance, au
   >>> 
   >>> sublimation = SublimationLTE(a, ice, rh)
   >>> print(sublimation.T[0], "K")  # doctest: +FLOAT_CMP
   160.52729461 K
   >>> print(sublimation.phi()[0], "kg/m2/s")  # doctest: +FLOAT_CMP
   1.35242694e-07 kg/m2/s


Grain sublimation lifetime
^^^^^^^^^^^^^^^^^^^^^^^^^^

Grain lifetimes may be calculated with the ``SublimationLTE.lifetime()`` method.  This requires a list of grain radii, so that it can integrate :math:`da/dt` from :math:`a_i` to :math:`a_0`.  Formally, for a grain to sublimate to :math:`a=0` approaches infinity.  In order to avoid this non-physical scenario, ``lifetime`` will sublimate the grain from :math:`a_0` to 0 using a constant radius loss rate (:math:`da/dt|a_0`).  Users must decide for themselves what to use for :math:`a_0`.

Calculate the lifetime of a 1.0 μm water ice grain at 1.0 au, with and without considerations for solar wind sputtering:

   >>> import numpy as np
   >>>
   >>> a = np.logspace(-2, 0)
   >>> sublimation = SublimationLTE(a, ice, rh)
   >>> tau = sublimation.lifetime()
   >>> print(tau[-1])  # doctest: +FLOAT_CMP
   8622.61 s


Thermal emission spectra
------------------------

Once the temperature of a grain is calculated, its spectrum may be generated using the absorption efficiencies (:math:`Q_{abs}`) and the grain temperature.  The following example calculates the spectrum of a 0.3 μm radius amorphous pyroxene grain (Mg/Fe=50/50) at 2 au from the Sun and 1 au from the observer, using the formula:

.. math::

   F_\nu = \frac{\pi a^2 Q_{abs} * B_{\nu}(T)}{\Delta^2}


First, setup our LTE instance.  Note that here we are tracking units with astropy, but the LTE classes do not accept astropy quantities, so we use `.value`:

   >>> import astropy.units as u
   >>> from mskpy.util import planck
   >>> from grains2 import ampyroxene50
   >>>
   >>> a = 0.3 * u.um
   >>> ap50 = ampyroxene50()
   >>> rh = 2.0 * u.au
   >>> delta = 1.0 * u.au
   >>>
   >>> lte = PlaneParallelIsotropicLTE(a.value, ap50, rh.value)

:math:`Q_{abs}` is a two-dimensional array.  The first dimension is for grain size, the second dimension is for wavelength:

   >>> lte.a.shape
   (1,)
   >>> lte.wave.shape
   (322,)
   >>> lte.Qabs.shape
   (1, 322)

Now, calculate the spectrum, limited to the 7 to 14 μm wavelength range:

   >>> i = (lte.wave > 7) * (lte.wave < 14)
   >>> wave = lte.wave[i]
   >>> Qabs = lte.Qabs[0, i]
   >>> B = planck(lte.T[0], wave, unit="Jy/sr")
   >>> F = (np.pi * a**2 * Qabs * B * u.sr / delta**2).to("Jy")

Plot the result:

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u
   from mskpy.util import planck
   from grains2 import PlaneParallelIsotropicLTE, ampyroxene50

   a = 0.3 * u.um
   ap50 = ampyroxene50()
   rh = 2.0 * u.au
   delta = 1.0 * u.au
   
   lte = PlaneParallelIsotropicLTE(a.value, ap50, rh.value)
   
   i = (lte.wave > 7) * (lte.wave < 14)
   wave = lte.wave[i]
   Qabs = lte.Qabs[0, i]
   B = planck(lte.T[0], wave, unit="Jy/sr")
   F = (np.pi * a**2 * Qabs * B * u.sr / delta**2).to("Jy")

   fig, ax = plt.subplots()
   ax.plot(wave, F, label="Amorphous pyroxene (Mg/Fe=50/50)")
   ax.legend()
   plt.setp(ax, xlabel="Wavelength (μm)", ylabel=r"$F_\nu$ (Jy)")




.. toctree::
   :maxdepth: 2
   :caption: Contents:

