Ice sublimation
===============

Ice sublimation is modeled with the ``SublimationLTE`` class, which balances absorbed sunlight with the energy losses from re-radiated thermal energy and the latent heat of sublimation of the ice.

The ``grains2`` water ice material includes the optical constants of water ice from Warren & Brandt (2008).  The latent heat of sublimation follows Delsemme & Miller (1971), and the vapor pressure equation of Lichtenegger & Komle (1991) is also used.


Temperature and mass-loss rate
------------------------------

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
--------------------------

Grain lifetimes may be calculated with the ``SublimationLTE.lifetime()`` method.  This requires a list of grain radii, so that it can integrate :math:`da/dt` from :math:`a_i` to :math:`a_0`.  Formally, for a grain to sublimate to :math:`a=0` approaches infinity.  In order to avoid this non-physical scenario, ``lifetime`` will sublimate the grain from :math:`a_0` to 0 using a constant radius loss rate (:math:`da/dt|a_0`).  Users must decide for themselves what to use for :math:`a_0`.

Calculate the lifetime of a 1.0 μm water ice grain at 1.0 au, with and without considerations for solar wind sputtering:

   >>> import numpy as np
   >>>
   >>> a = np.logspace(-2, 0)
   >>> sublimation = SublimationLTE(a, ice, rh)
   >>> tau = sublimation.lifetime()
   >>> print(tau[-1])  # doctest: +FLOAT_CMP
   8622.61 s
