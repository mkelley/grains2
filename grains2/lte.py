"""
lte --- Grains in LTE
=====================

Classes
-------
RadiativeLTE
PlaneParallelIsotropicLTE
SublimationLTE

"""

from abc import ABC, abstractmethod
import numpy as np
from numpy import pi
from scipy import optimize
from scipy.interpolate import splrep, splev

import astropy.units as u
from mskpy import calib
from mskpy.util import planck
from mskpy.util import autodoc

from .material import Material, Ice
from .scattering import ScatteringModel, Mie
from .porosity import PorosityModel, Solid
from .davint import davint


__all__ = ["PlaneParallelIsotropicLTE", "SublimationLTE"]


class RadiativeLTE(ABC):
    """Abstract base class for grains in radiative LTE.

    The names of inheriting classes should specify the geometry of the problem
    in two parts: the geometry of the incident radiation followed by the
    geometry of the emitted radiation.

    Grain radius and material must be the first two arguments for
    initialization.

    Parameters
    ----------
    a : float or array
        Grain radii.

    m : Material
        The grain material.

    scattering : ScatteringModel, optional
        Light scattering model with which to compute LTE.

    S : array or string, optional
        2xN array specifying the wavelengths and flux densities of the incident
        radiation [μm, W/m2/μm], or one of the following strings (case
        in-sensitive):

            * "Wehrli": Wehrli 1985 solar spectrum at 1 AU (smoothed)

            * "E490": E490 solar spectrum at 1 AU (smoothed)

            * "harker mie idl": the spectrum used in the Harker et al. papers.

    wave : ndarray, optional
        Use these specific wavelengths when computing Qabs, temperature, etc. If
        set to None, ``RadiativeLTE`` will concatenate the wavelengths from
        ``S`` with the wavelengths from ``m.relref().wave``.

    update : bool, optional
        Set to True to update on initialization.

    """

    def __init__(
        self, a, m, scattering=None, porosity=None, S="E490", wave=None, update=True
    ):

        self._T = np.array([0])
        self._updated = False
        self._updating = False

        self.a = np.array(a)
        if self.a.ndim == 0:
            self.a = np.array([a])

        self.m = m
        assert isinstance(self.m, Material)

        self.scattering = Mie() if scattering is None else scattering
        assert isinstance(self.scattering, ScatteringModel)

        self.porosity = Solid() if porosity is None else porosity
        assert isinstance(self.porosity, PorosityModel)

        if type(S) == str:
            unit = u.Unit("W/(m2 um)")
            if S.lower() == "wehrli":
                w, f = np.array(calib.wehrli(smooth=True, unit=unit))[:, :-1]
                S = (w.to(u.um).value, f.value)
            elif S.lower() == "e490":
                w, f = calib.e490(smooth=True, unit=unit)
                S = (w.to(u.um).value, f.value)
            elif S.lower() == "harker mie idl":
                # solar spectrum used by D. Harker, dated 19 Mar 1999
                temp = [
                    0.20,
                    0.22,
                    0.24,
                    0.26,
                    0.28,
                    0.30,
                    0.32,
                    0.34,
                    0.36,
                    0.37,
                    0.38,
                    0.39,
                    0.40,
                    0.41,
                    0.42,
                    0.43,
                    0.44,
                    0.45,
                    0.46,
                    0.48,
                    0.50,
                    0.55,
                    0.60,
                    0.65,
                    0.70,
                    0.75,
                    0.80,
                    0.90,
                    1.0,
                    1.1,
                    1.2,
                    1.4,
                    1.6,
                    1.8,
                    2.0,
                    2.5,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    8.0,
                    10.0,
                    12.0,
                ]
                S = np.zeros((2, len(temp)))
                S[0] = temp
                temp = [
                    1.2,
                    4.5,
                    6.4,
                    13.0,
                    25.0,
                    59.0,
                    85.0,
                    114.0,
                    115.0,
                    127.0,
                    121.0,
                    115.0,
                    160.0,
                    187.0,
                    189.0,
                    183.0,
                    201.0,
                    213.0,
                    215.0,
                    213.0,
                    204.0,
                    198.0,
                    187.0,
                    167.0,
                    149.0,
                    129.0,
                    114.0,
                    90.0,
                    74.0,
                    61.0,
                    50.0,
                    33.0,
                    22.3,
                    14.8,
                    10.2,
                    4.97,
                    2.63,
                    0.93,
                    0.41,
                    0.21,
                    0.063,
                    0.023,
                    0.012,
                ]
                S[1] = np.array(temp) * 10.0
                del temp
            else:
                raise ValueError("Unknown S: {}".format(S))
        if len(S) != 2:
            raise ValueError("S has incorrect shape; should be 2xN.")

        Sw = S[0]
        Sf = S[1] / self.r**2

        self._prepareSRQ(Sw, Sf, wave=wave)
        if update:
            self.update()

    def _prepareSRQ(self, Sw, Sf, wave=None):
        """Prepare variables for numeric integration.

        solar spec., refractive index, and Qabs.

        """

        if wave is None:
            self.wave = np.sort(np.concatenate((Sw, self.m.ri.wave)))
            self.wave = np.unique(self.wave.round(decimals=4))
        else:
            self.wave = wave

        # 0 outside interpolation range
        S = splrep(np.log(Sw), np.log(Sf))
        v = splev(np.log(self.wave), S, ext=1)
        i = v != 0
        self.S = np.zeros_like(self.wave)
        self.S[i] = np.exp(v[i])

        self.R = np.zeros((len(self.a), self.wave.size), complex)
        self.Qabs = np.zeros((len(self.a), self.wave.size))
        self.Qsca = np.zeros((len(self.a), self.wave.size))
        self.Qext = np.zeros((len(self.a), self.wave.size))
        self.SQ = np.zeros_like(self.a)
        SQsca = np.zeros_like(self.a)
        SQext = np.zeros_like(self.a)
        self.Qabs_bar = np.zeros_like(self.a)
        self.Qsca_bar = np.zeros_like(self.a)
        self.Qext_bar = np.zeros_like(self.a)

        self.Qabs_orig = np.zeros((len(self.a), self.m.ri.wave.size))
        self.Qsca_orig = np.zeros((len(self.a), self.m.ri.wave.size))
        self.Qext_orig = np.zeros((len(self.a), self.m.ri.wave.size))
        for i in range(len(self.a)):
            # compute Qabs, Qsca, and Qext, then interpolate
            w = self.m.ri.wave
            m = self.porosity.apply(self.m, self.a[i])
            Q = self.scattering.q(self.a[i], w, m.ri)

            self.Qabs_orig[i] = Q["qabs"]
            self.Qsca_orig[i] = Q["qsca"]
            self.Qext_orig[i] = Q["qext"]

            j = Q["qabs"] > 0
            v = splev(
                np.log(self.wave), splrep(np.log(w[j]), np.log(Q["qabs"][j])), ext=1
            )
            j = v != 0
            self.Qabs[i, j] = np.exp(v[j])

            if any(self.Qabs[i] < 0):
                print("Warning: Found Qabs value(s) less than zero.  Fixing.")
                k = self.Qabs[i] < 0
                print(self.wave[k])
                self.Qabs[i, k] = 0

            j = Q["qsca"] > 0
            v = splev(
                np.log(self.wave), splrep(np.log(w[j]), np.log(Q["qsca"][j])), ext=1
            )
            j = v != 0
            self.Qsca[i, j] = np.exp(v[j])

            if any(self.Qsca[i] < 0):
                print("Warning: Found Qsca value(s) less than zero.  Fixing.")
                k = self.Qsca[i] < 0
                print(self.wave[k])
                self.Qsca[i, k] = 0

            j = Q["qext"] > 0
            v = splev(
                np.log(self.wave), splrep(np.log(w[j]), np.log(Q["qext"][j])), ext=1
            )
            j = v != 0
            self.Qext[i, j] = np.exp(v[j])

            if any(self.Qext[i] < 0):
                print("Warning: Found Qext value(s) less than zero.  Fixing.")
                k = self.Qext[i] < 0
                print(self.wave[k])
                self.Qext[i, k] = 0

            j = (self.S * self.Qabs[i]) > 0
            self.SQ[i] = davint(
                self.wave[j],
                self.S[j] * self.Qabs[i, j],
                self.wave[j][0],
                self.wave[j][-1],
            )
            temp = davint(self.wave[j], self.S[j], self.wave[j][0], self.wave[j][-1])
            self.Qabs_bar[i] = self.SQ[i] / temp

            j = (self.S * self.Qsca[i]) > 0
            SQsca[i] = davint(
                self.wave[j],
                self.S[j] * self.Qsca[i, j],
                self.wave[j][0],
                self.wave[j][-1],
            )
            self.Qsca_bar[i] = SQsca[i] / temp

            j = (self.S * self.Qext[i]) > 0
            SQext[i] = davint(
                self.wave[j],
                self.S[j] * self.Qext[i, j],
                self.wave[j][0],
                self.wave[j][-1],
            )
            self.Qext_bar[i] = SQext[i] / temp

    @property
    def T(self):
        """Grain physical temperature.  [K]"""
        if not self._updated and not self._updating:
            raise RuntimeError("Temperature not up to date.")
        return self._T

    def Eabs(self):
        """Absorbed energy. [W]"""
        E = np.zeros(np.size(self.a))
        for i in range(E.shape[0]):
            E[i] = self._Eabs(self.a[i], self.SQ[i])
        if len(E) == 1:
            return E[0]
        else:
            return E

    def Erad(self, T=None):
        """Radiated energy at temperature T.


        Parameters
        ----------
        T : ndarray or float
            If a float, 1 temperature will be used for all a.  [K]


        Returns
        -------
        E : ndarray or float
            Radiated energy. [W]

        """

        T = self.T if T is None else T
        if np.size(T) == 1:
            T = np.ones_like(self.a) * T

        E = np.zeros(np.size(self.a))
        for i in range(E.shape[0]):
            E[i] = self._Erad(self.a[i], self.Qabs[i], T[i])

        if len(E) == 1:
            return E[0]
        else:
            return E

    @abstractmethod
    def _Eabs(self, a, SQ):
        """Energy absorbed.

        Scalar function.

        """
        pass

    @abstractmethod
    def _Erad(self, a, Qabs, T):
        """Energy radiated.

        Scalar function.

        """
        pass

    def dE(self, T=None):
        """Difference between E_abs and E_em at temperature T.


        Parameters
        ----------
        T : ndarray or float
            If a float, 1 temperature will be used for all a.  [K]


        Returns
        -------
        dE : ndarray or float
            Energy absorbed - energy lost. [W]

        """
        T = self.T if T is None else T
        if np.size(T) == 1:
            T = np.ones_like(self.a) * T

        dE = np.zeros(np.size(self.a))
        for i in range(dE.shape[0]):
            dE[i] = self._conserveEnergy(self.a[i], self.SQ[i], self.Qabs[i], T[i])

        if len(dE) == 1:
            return dE[0]
        else:
            return dE

    def _conserveEnergy_brentq(self, T, i):
        """For optimize.brentq (T will be scalar).  When self.a is a
        scalar, pass i = None."""
        if i is None:
            dE = self._conserveEnergy(self.a, self.SQ[0], self.Qabs[0], T)
        else:
            dE = self._conserveEnergy(self.a[i], self.SQ[i], self.Qabs[i], T)
        return dE

    def _conserveEnergy(self, a, SQ, Qabs, T):
        """Scalar function."""
        return self._Eabs(a, SQ) - self._Erad(a, Qabs, T)

    def update(self, T0=20, T1=2000):
        """Find the temperature for each a in LTE."""

        if self._updated:
            raise ValueError("Solution already up to date.")

        # conserve energy
        self._updating = True

        # solve for each size
        self._T, self.BQ, self.Qrad_bar = np.zeros((3, len(self.a)))
        for i in range(len(self.a)):
            self._T[i] = optimize.brentq(
                self._conserveEnergy_brentq, T0, T1, args=(i,), xtol=1e-10
            )
            # update Qrad_bar with the solution
            B = pi * planck(self.wave, self._T[i], unit=u.Unit("W/(m2 sr um)")).value
            self.BQ[i] = davint(
                self.wave, B * self.Qabs[i], self.wave[0], self.wave[-1]
            )
            temp = davint(self.wave, B, self.wave[0], self.wave[-1])
            self.Qrad_bar[i] = self.BQ[i] / temp

        self._updated = True
        self._updating = False


class PlaneParallelIsotropicLTE(RadiativeLTE):
    """Plane parallel incident radiation, isotropic thermal emission.

    Parameters
    ----------
    a : float or array
        Grain radii.

    m : Material
        The grain material.

    r : float
        Relavtive distance to the radiation source, i.e., scale ``S`` by
        ``1/r**2``.  Typically, ``S`` will be solar flux at 1 AU, and ``r`` will
        be heliocentric distance in AU.

    scattering : ScatteringModel, optional
        Light scattering model with which to compute LTE.

    S : array or string, optional
        2xN array specifying the wavelengths and flux densities of the incident
        radiation [μm, W/m2/μm], or one of the following strings (case
        in-sensitive):

            * "Wehrli": Wehrli 1985 solar spectrum at 1 AU (smoothed)

            * "E490": E490 solar spectrum at 1 AU (smoothed)

            * "harker mie idl": the spectrum used in the Harker et al. papers.

    wave : ndarray, optional
        Use these specific wavelengths when computing Qabs, temperature, etc. If
        set to None, ``RadiativeLTE`` will concatenate the wavelengths from
        ``S`` with the wavelengths from ``m.relref().wave``.

    update : bool, optional
        Set to True to update on initialization.

    """

    def __init__(
        self, a, m, r, scattering=None, porosity=None, S="E490", wave=None, update=True
    ):
        self.r = r
        assert isinstance(self.r, float)

        super().__init__(
            a,
            m,
            scattering=scattering,
            porosity=porosity,
            S=S,
            wave=wave,
            update=update,
        )

    def __repr__(self):
        return """<PlaneParallelIsotropicLTE for {} grains
  r = {}
  a = {}
  Qabs = {}
  Qabs_bar = {}
  T = {}
  Qrad_bar = {}
>""".format(
            self.m.name,
            self.r,
            np.array2string(self.a, 74, prefix="  "),
            np.array2string(self.Qabs, 71, prefix="  "),
            np.array2string(self.Qabs_bar, 67, prefix="  "),
            np.array2string(np.array(self.T), 74, prefix="  "),
            np.array2string(self.Qrad_bar, 67, prefix="  "),
        )

    def _Eabs(self, a, SQ):
        return pi * (a * 1e-6) ** 2 * SQ

    def _Erad(self, a, Qabs, T):
        B = pi * planck(self.wave, T, unit=u.Unit("W/(m2 sr um)")).value
        BQ = davint(self.wave, B * Qabs, self.wave[0], self.wave[-1])
        return 4.0 * pi * (a * 1e-6) ** 2 * BQ


class SublimationLTE(PlaneParallelIsotropicLTE):
    lines = PlaneParallelIsotropicLTE.__doc__.splitlines()[1:]
    i = lines.index("    m : Material")
    __doc__ = (
        "Plane parallel incident radiation, isotropic thermal emission, isotropic sublimation."
        + "\n".join(lines[:i])
        + "    m : Ice\n"
        + "\n".join(lines[i + 1 :])
    )

    def __init__(self, a, m, rh, **kwargs):
        if not isinstance(m, Ice):
            raise ValueError(f"Expecting an Ice, got {type(m)}")
        super().__init__(a, m, rh, **kwargs)

    def _Erad(self, a, Qabs, T):
        return super()._Erad(a, Qabs, T) + self._Esubl(a, T)

    def _Esubl(self, a, T):
        """Scalar function."""
        return 4.0 * pi * (a * 1e-6) ** 2 * self.m.H(T) * self.phi(T)

    def phi(self, T=None):
        """Sublimation rate.  [kg/m2/s]"""
        kB = 1.3806488e-23  # J/K
        T = self.T if T is None else T
        if not self._updated and not self._updating:
            self.update()
        return self.m.Pv(T) * np.sqrt(self.m.mu / (2.0 * pi * kB * T))

    def Q(self, T=None):
        """Production rate.  [kg/s]"""
        T = self.T if T is None else T
        if np.size(T) == 1:
            T = np.ones_like(self.a) * T
        return 4.0 * pi * (self.a * 1e-6) ** 2 * self.phi(T)

    def Qmps(self, T=None):
        """Production rate.  [molec/s]"""
        return self.Q(T) / self.m.mu

    def Z(self, T=None):
        """Sublimation rate.  [molec/cm2/s]"""
        return self.phi(T) / self.m.mu * 1e-4

    def dadt(self, T=None):
        """Differential variation of radius with time. [micron/s]"""
        T = self.T if T is None else T
        return -1e6 * self.phi(T) / (self.m.rho * 1e3)

    def lifetime(self, sputtering=True):
        """Estimate grain lifetimes.

        Sublimation rate is a function of grain size.  Integrating the rate
        formally produces infinite lifetimes.  Here, the sublimation rate is
        integrated for all grain sizes, but for the last two sizes
        (``self.a[:1]``), a linear sublimation rate with time is assumed, which
        allows for the grain to shrink to size 0.


        Parameters
        ----------
        sputtering : bool, optional
            ``False`` to exclude solar wind sputtering.


        Returns
        -------
        tau : Quantity
            Total lifetime for grains of size ``self.a`` to decrease down to
            size 0.

        """

        dtda = self.dadt() ** -1

        if sputtering:
            if not hasattr(self.m, "sputtering"):
                raise ValueError(f"{self.m} does not have a sputtering function")
            dtda = (
                -1e7 * self.m.mu / self.m.rho * self.m.sputtering(self.r).value
                + dtda**-1
            ) ** -1

        tau = np.zeros(self.a.shape)
        # davint requires 3 values between integration limits
        for i in range(2, len(self.a)):
            tau[i] = davint(self.a, -dtda, self.a[0], self.a[i])
        tau[1] = (self.a[1] - self.a[0]) * -dtda[0]
        tau += -self.a[0] * dtda[0]

        return tau * u.s


# update module docstring
autodoc(globals())
del autodoc
