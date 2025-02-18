"""
emission --- Thermal emission from dust
=======================================

Classes
-------
Coma
CometDust
Harker07

"""

import numpy as np
from numpy import pi
from scipy.interpolate import splrep, splev
import astropy.units as u
from mskpy.util import planck, minmax

from .lte import PlaneParallelIsotropicLTE
from . import material as mat
from .scattering import ScatteringModel, Mie, ProlateCDE
from .porosity import Solid
from .gsd import GSD
from .davint import davint


__all__ = ["Coma", "CometDust", "Harker07"]


class Coma(dict):
    """A collection of `CometDust`s.

    Parameters
    ----------
    delta : Quantity, optional
        The observer-coma distance.

    **kwargs :
        ``CometDust`` components.

    Example
    -------
    >>> a = np.logspace(-1, 2, 100)
    >>> gsd = gsd.Hanner(0.1, 3.7, ap=0.4)
    >>> coma = Coma()
    >>> for k, m in zip(('ac', 'ao'), (amcarbon(), amolivine50())):
    >>>     coma[k] = CometDust(a, m, gsd, rh=1.3)

    """

    def __init__(self, dust=(), delta=1.0):
        for d in dust:
            self.append(d)

        self.delta = u.Quantity(delta, u.au)

    def fluxd(self, wave, unit="W/(m2 um)"):
        """Compute thermal emission spectrum from this coma.

        The dust is weighted by the grain size distribution and the
        total flux density returned.


        Parameters
        ----------
        wave : Quantity
            Wavelengths at which to compute thermal emission.

        unit : astropy Unit, optional
            Output flux density units.


        Returns
        -------
        dict of ndarrays

        """
        wave = u.Quantity(wave, u.um)
        f = dict()
        for k, v in self.items():
            f[k] = v.fluxd(self.delta, wave, unit=unit, sum=True)
        return f

    def update(self, gsd=None):
        pass


class CometDust(PlaneParallelIsotropicLTE):
    """A comet coma in LTE with solar insolation.


    Parameters
    ----------
    rh : float
        The distance to the Sun. [AU]

    a : float or array
        Grain radii.  [micron]

    m : Material
        Grain material.

    gsd : GSD, optional
        Grain size distribution or ``None``.  If ``None``, ``fluxd()`` will return
        fluxes for each size ``a``.

    porosity : PorosityModel, optional
        Grain porosity model.

    scattering : ScatteringModel, optional
        Light scattering model with which to compute LTE.

    spec_model : ScatteringModel, optional
        Light scattering model with which to compute emission spectrum, or
        ``None`` to use ``scattering``.

    tscale : float
        Scale grain equilibrium temperatures by this factor before computing
        thermal emission.

    **kwargs
        Any ``PlaneParallelIsotropicLTE`` keyword, e.g., ``S``, ``wave``,
        ``update``.


    """

    def __init__(
        self,
        rh,
        a,
        m,
        gsd=None,
        porosity=None,
        scattering=None,
        spec_model=None,
        tscale=1.0,
        **kwargs
    ):
        self.spec_model = spec_model
        if self.spec_model is not None:
            assert isinstance(self.spec_model, ScatteringModel)

        PlaneParallelIsotropicLTE.__init__(
            self, rh, a, m, scattering=scattering, porosity=porosity
        )

        self.gsd = gsd
        if self.gsd is not None:
            assert isinstance(self.gsd, GSD)

        self.tscale = tscale

    def __repr__(self):
        return """<CometDust with {} grains
  rh = {}
  a = {}
  Qabs = {}
  Qabs_bar = {}
  Qabs_spec = {}
  T = {}
  Qrad_bar = {}
>""".format(
            self.m.name,
            self.rh,
            np.array2string(self.a, 74, prefix="  "),
            np.array2string(self.Qabs, 71, prefix="  "),
            np.array2string(self.Qabs_bar, 67, prefix="  "),
            np.array2string(self.Qabs_spec, 67, prefix="  "),
            np.array2string(np.array(self.T), 74, prefix="  "),
            np.array2string(self.Qrad_bar, 67, prefix="  "),
        )

    @property
    def rh(self):
        return self.r

    def fluxd(self, delta, wave, unit="W/(m2 um)", sum=True):
        """Compute thermal emission spectrum from this dust.

        The if `gsd` is not `None`, the dust is weighted by the grain size
        distribution and the total is returned.


        Parameters
        ----------
        delta : Quantity
            Distance to comet.

        wave : Quantity
            Wavelengths at which to compute thermal emission.

        unit : astropy Unit, optional
            Output flux density units.

        sum : bool, optional
            Set to ``False`` to return each grain's spectrum, unweighted by the
            grain size distribution.


        Returns
        -------
        ndarray

        """

        w = u.Quantity(wave, u.um).value
        w = np.array(w) if np.iterable(w) else np.array([w])
        unit = u.Unit(unit)

        assert self._updated, "Temperature not up to date."
        D2 = delta.to(u.um).value ** 2
        f = np.ones((len(self.a), len(w))) * unit
        for i in range(len(self.a)):
            B = planck(w, self.T[i] * self.tscale, unit=unit / u.sr)
            Qabs = splev(w, splrep(self.wave, self.Qabs_spec[i]), ext=1)
            f[i] = B * pi * u.sr * self.a[i] ** 2 * Qabs / D2

        if not sum or self.gsd is None:
            return f

        if len(self.a) == 1:
            F = (self.gsd.dnda(self.a) * f)[0]
        else:
            F = davint(self.a, self.gsd.dnda(self.a) * f.T, self.a[0], self.a[-1])
        return F

    def update(self, **keywords):
        PlaneParallelIsotropicLTE.update(self, **keywords)
        if self.spec_model is not None:
            self.Qabs_spec = self.spec_model.qabs(self.a, self.wave, self.m.ri)
        else:
            self.Qabs_spec = self.Qabs


class Harker07(Coma):
    """Comet coma based on Harker et al. 2007.

    Parameters
    ----------
    rh : Quantity
        Comet heliocentric distance.

    delta : Quanitity
        Comet-observer distance.

    gsd : GSD
        A single grain size distribution for all dust components.

    porosity : PorosityModel
        A single porosity model for all amorphous dust components. Crystalline
        silicates are always solid.

    arange : Quantity, optional
        Limit grain sizes to ``min(arange)``, ``max(arange)``.

    arange_cryst : Quantity, optional
        Limit crystalline grain sizes to this range.

    """

    _num_per_decade = 20

    def __init__(
        self, rh, delta, gsd, porosity, arange=[0.1, 100], arange_cryst=[0.1, 1.0]
    ):
        print("WARNING: Ortho-pyroxene not yet implemented.")

        self._rh = u.Quantity(rh, u.au)
        self._delta = u.Quantity(delta, u.au)

        self._gsd = gsd
        self._porosity = porosity
        self._arange = minmax(u.Quantity(arange, u.um).value)
        self._arange_cryst = minmax(u.Quantity(arange_cryst, u.um).value)

        self._load_dust()

    def _load_dust(self):
        x = np.log10(self.arange)
        a = np.logspace(x[0], x[1], x.ptp() * 20)
        print("Loading amorphous components")
        print("  - carbon")
        self["ac"] = CometDust(
            self.rh.value,
            a,
            mat.amcarbon(),
            self.gsd,
            porosity=self.porosity,
            scattering=Mie(),
        )
        print("  - olivine")
        self["ao"] = CometDust(
            self.rh.value,
            a,
            mat.amolivine50(),
            self.gsd,
            porosity=self.porosity,
            scattering=Mie(),
        )
        print("  - pyroxene")
        self["ap"] = CometDust(
            self.rh.value,
            a,
            mat.ampyroxene50(),
            self.gsd,
            porosity=self.porosity,
            scattering=Mie(),
        )

        x = np.log10(self.arange_cryst)
        a = np.logspace(x[0], x[1], x.ptp() * self._num_per_decade)
        print("Loading crystalline components")
        print("  - Mg-rich olivine")
        self["co"] = CometDust(
            self.rh.value,
            a,
            mat.olivine95(),
            self.gsd,
            porosity=Solid(),
            scattering=Mie(),
            spec_model=ProlateCDE("z", 10),
            tscale=1.9,
        )

    @property
    def arange(self):
        return self._arange

    @property
    def arange_cryst(self):
        return self._arange_cryst

    @property
    def delta(self):
        return self._delta

    @property
    def gsd(self):
        return self._gsd

    @property
    def porosity(self):
        return self._porosity

    @property
    def rh(self):
        return self._rh

    def update(**kwargs):
        """Update the model parameters.

        Parameters
        ----------
        rh : Quantity, optional
        delta : Quantity, optional
        gsd : GSD, optional
        porosity : PorosityModel, optional
        arange : Quantity, optional
        arange_cryst : Quantity, optional

        """

        update_amorphous = False
        update_crystals = False

        delta = kwargs.pop("delta", None)
        if delta is not None:
            self.delta = u.Quantity(delta, u.au).value

        arange = kwargs.pop("arange", None)
        if arange is not None:
            update_amorphous = True
            x = np.log10(self.arange)
            a = np.logspace(x[0], x[1], x.ptp() * self._num_per_decade)
            for k in ["ac", "ao", "ap"]:
                self[k].a = a

        arange_cryst = kwargs.pop("arange_cryst", None)
        if arange_cryst is not None:
            update_crystals = True
            x = np.log10(self.arange_cryst)
            a = np.logspace(x[0], x[1], x.ptp() * self._num_per_decade)
            for k in ["co"]:
                self[k].a = a

        porosity = kwargs.pop("porosity", None)
        if porosity is not None:
            update_amorphous = True
            for k in ["ac", "ao", "ap"]:
                self[k].porosity = porosity

        rh = kwargs.pop("rh", None)
        if rh is not None:
            update_amorphous = True
            update_crystals = True
            for k in ["ac", "ao", "ap", "co"]:
                self[k].r = rh

        gsd = kwargs.pop("gsd", None)
        if gsd is not None:
            for k in ["ac", "ao", "ap", "co"]:
                self[k].gsd = gsd

        if update_amorphous:
            print("Updating amorphous dust")
            for k in ["ac", "ao", "ap"]:
                self[k].update()

        if update_crystals:
            print("Updating crystalline dust")
            for k in ["co"]:
                self[k].update()


# update module docstring
from mskpy.util import autodoc

autodoc(globals())
del autodoc
