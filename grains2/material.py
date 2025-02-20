"""
material --- Grain materials
============================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   RefractiveIndices
   Material

   Functions
   ---------
   load_refractive_indices

   amcarbon
   amolivine40
   amolivine50
   ampyroxene40
   ampyroxene50
   ferropericlase50
   magnetite
   neutral
   olivine95
   vacuum
   waterice
   wustite

"""

from importlib import resources

import numpy as np
from scipy.interpolate import splrep, splev

import astropy.units as u
from astropy.io import ascii
from astropy.table import Table, Column

__all__ = [
    "amcarbon",
    "amolivine40",
    "amolivine50",
    "ampyroxene40",
    "ampyroxene50",
    "magnetite",
    "neutral",
    "olivine95",
    "vacuum",
    "waterice",
]


class RefractiveIndices:
    """Refractive index.


    Parameters
    ----------
    wave : ndarray
        The wavelengths. [micron]

    nk : complex ndarray, or dictionary
        The indices of refraction.  May be a dictionary of N-element arrays to
        specify different axes for anisotripic materials (see Examples).

    Attributes
    ----------
    wave : array
        The wavelengths. [micron]

    nk : array
        The complex indices of refraction.

    n : array
        The real part of nk.

    k : array
        The imaginary part of nk.


    Examples
    --------
    Anisotripic materials can be specified, but attributes
    (``RefractiveIndices.n``) will return the mean of all axes. Individual axes
    can be returned by indexing the ``RefractiveIndices`` object:

    >>> from grains2.material import RefractiveIndices as RI
    >>> w = np.linspace(0.3, 0.6, 5)
    >>> nk = np.ones((3, len(w)), complex)
    >>> nk[1] *= 1.22
    >>> nk[2] *= 1.5 + 0.03j
    >>> ri = RI(w, dict(x=nk[0], y=nk[1], z=nk[2]))
    >>> print(ri.nk)
    [1.24+0.01j 1.24+0.01j 1.24+0.01j 1.24+0.01j 1.24+0.01j]
    >>> print(ri['y'].nk)
    [1.22+0.j 1.22+0.j 1.22+0.j 1.22+0.j 1.22+0.j]

    """

    def __init__(self, wave, nk):
        self.wave = wave
        self.nk = nk
        self.anisotropic = False

        if isinstance(self.nk, dict):
            # multiple indices were given
            self.anisotropic = True
            self.axes = self.nk
            self.nk = np.array(list(self.axes.values()), complex).mean(0)
        else:
            self.nk = np.array(self.nk)
            self.axes = dict()

        if len(self.wave) != len(self.nk):
            raise ValueError("The lengths of wave and nk are not equal.")

    def __getitem__(self, key):
        if key in self.axes:
            return RefractiveIndices(wave=self.wave, nk=self.axes[key])
        else:
            raise KeyError

    def __call__(self, wave, log=True):
        """Interpolate nk onto new wavelengths.


        Parameters
        ----------
        wave : array or Quantity
            The requested wavelengths.

        log : bool, optional
            Set to True to interpolate ``k`` in logspace.


        Returns
        -------
        nk : array
            The interpolated values, or 0 if not defined at ``wave``.

        """

        wave = u.Quantity(wave, u.um).value

        n = splrep(self.wave, self.n)
        if log:
            k = splrep(self.wave, np.log(self.k))
            newk = splev(wave, k, ext=1)
            if newk.size > 1:
                i = (newk != 0) * np.isfinite(newk)
                newk[i] = np.exp(newk[i])
                newk[~i] = 0
            else:
                if (newk != 0) and np.isfinite(newk):
                    newk = np.exp(newk)
                else:
                    newk = 0
            return splev(wave, n, ext=1) + newk * 1j
        else:
            k = splrep(self.wave, self.k)
            return splev(wave, n, ext=1) + splev(wave, k, ext=1) * 1j

    @property
    def n(self):
        return np.real(self.nk)

    @property
    def k(self):
        return np.imag(self.nk)

    def copy(self):
        """Return a copy of this object."""
        if len(self.axes) > 0:
            return RefractiveIndices(self.wave, self.axes)
        else:
            return RefractiveIndices(self.wave, self.nk)

    def items(self):
        """Keys and values of the anisotropic axes."""
        return self.axes.items()

    def keys(self):
        """Names of the anisotropic axes."""
        return self.axes.keys()

    def save(self, filename):
        """Save refractive indices to a file.


        Parameters
        ----------
        filename : string
            The name of the file.

        """

        self.table().write(filename, format="ascii.fixed_width_two_line")

    def table(self):
        """The refractive indices as a table."""

        t = Table()
        t.add_column(Column(name="wave", data=self.wave))
        if len(self.axes) > 0:
            for k, v in self.axes:
                t.add_column(Column(name=k, data=v))
        else:
            t.add_column(Column(name="nk", data=self.nk))

        return t

    def values(self):
        """nk for each of the anisotropic axes."""
        return self.axes.values()


def load_refractive_indices(filename):
    """Load a saved set of refractive indices.


    Parameter
    ---------
    filename : string
        The name of the file.


    Returns
    -------
    RefractiveIndices

    """
    t = ascii.read(filename, format="fixed_width_two_line")
    if len(t.colnames) == 2:
        return RefractiveIndices(t["wave"].data, t["nk"].data)
    else:
        axes = dict()
        for k in t.colnames[1:]:
            axes[k] = t[k].data
        return RefractiveIndices(t["wave"].data, axes)


class Material:
    """Physical properties of materials.

    Parameters
    ----------
    name : string
        A string name of the material.

    rho : float, optional
        The bulk density.  [g/cm3]

    ri : RefractiveIndices, optional
        The refractive indices.

    mu : float, optional
        The mass of one molecule (relevant for ices). [kg]

    Pv : function, optional
        Vapor pressure at temperature ``T`` in K: Pv(T).  [N/m2]

    H : function, optional
        Latent heat of sublimation at temperature ``T`` in K: H(T). [J/kg]

    """

    def __init__(self, name, **keywords):
        self.name = name
        self.rho = keywords.get("rho")
        self.ri = keywords.get("ri")
        self.mu = keywords.get("mu")
        self.Pv = keywords.get("Pv", lambda T: None)
        self.H = keywords.get("H", lambda T: None)

    def __repr__(self):
        return "<Material [{}]>".format(self.name)


class Ice(Material):
    """Physical properties of a volatile material (ice).

    Parameters
    ----------
    name : string
        A string name of the material.

    rho : float
        The bulk density.  [g/cm3]

    ri : RefractiveIndices
        The refractive indices.

    mu : float
        The mass of one molecule (relevant for ices). [kg]

    Pv : function
        Vapor pressure at temperature ``T`` in K: Pv(T).  [N/m2]

    H : function
        Latent heat of sublimation at temperature ``T`` in K: H(T). [J/kg]

    sputtering : function
        Solar wind sputtering rate.  []

    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.rho = kwargs.get("rho")
        self.ri = kwargs.get("ri")
        self.mu = kwargs.get("mu")
        self.Pv = kwargs.get("Pv", lambda T: None)
        self.H = kwargs.get("H", lambda T: None)
        self.sputtering = kwargs.get("sputtering", None)

    def __repr__(self):
        return "<Ice [{}]>".format(self.name)


def amcarbon(**keywords):
    """Amorphous carbon."""
    ref = resources.files("grains2") / "data/am-carbon.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous carbon", rho=1.5, ri=ri, **keywords)


def amolivine40(**keywords):
    """ "Amorphous" olivine with Mg/Fe = 40/60."""
    ref = resources.files("grains2") / "data/olivine-mg40.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous olivine 40/60", rho=3.3, ri=ri, **keywords)


def amolivine50(**keywords):
    """ "Amorphous" olivine with Mg/Fe = 50/50."""
    ref = resources.files("grains2") / "data/olivine-mg50.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous olivine 50/50", rho=3.3, ri=ri, **keywords)


def ampyroxene40(**keywords):
    """ "Amorphous" pyroxene with Mg/Fe = 40/60."""
    ref = resources.files("grains2") / "data/pyroxene-mg40.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous pyroxene 40/60", rho=3.3, ri=ri, **keywords)


def ampyroxene50(**keywords):
    """ "Amorphous" pyroxene with Mg/Fe = 50/50."""
    ref = resources.files("grains2") / "data/pyroxene-mg50.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous pyroxene 50/50", rho=3.3, ri=ri, **keywords)


def magnetite(**keywords):
    """Magnetite, Fe3O4."""
    ref = resources.files("grains2") / "data/magnetite.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[:, 0], x[:, 1] + 1j * x[:, 2])
    return Material("magnetite", ri=ri, **keywords)


def wustite(**keywords):
    """Wüstite, FeO."""
    ref = resources.files("grains2") / "data/fe100o.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn, skiprows=4).T
    ri = RefractiveIndices(x[:, 0], x[:, 1] + 1j * x[:, 2])
    return Material("Wüstite", ri=ri, **keywords)


def ferropericlase50(**keywords):
    """Ferropericlase Mg_0.5 Fe_0.5 O."""
    ref = resources.files("grains2") / "data/fe50o.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn, skiprows=4).T
    ri = RefractiveIndices(x[:, 0], x[:, 1] + 1j * x[:, 2])
    return Material("ferropericlase 50", ri=ri, **keywords)


def neutral(nk, wave=None, **keywords):
    """A neutral scatterer.


    Parameters
    ----------
    nk : complex
        The refractive index for all wavelengths.

    wave : array, optional
        Use these wavelengths for creating the indices of refraction.

    """
    if wave is None:
        wave = np.logspace(-2, 3, 1000)
    ri = RefractiveIndices(wave, np.ones(len(wave), complex) * nk)
    return Material("neutral scattering", ri=ri, **keywords)


def olivine95(**keywords):
    """Olivine with Mg/Fe = 95/5 (Mg_1.9 Fe_0.1 SiO_4)."""
    ref = resources.files("grains2") / "data/oliv_vis.txt"
    with resources.as_file(ref) as fn:
        vis = np.loadtxt(fn).T

    nk = dict()
    for i in "xyz":
        ref = resources.files("grains2") / f"data/oliv_nk_{i}.txt"
        with resources.as_file(ref) as fn:
            ir = np.loadtxt(fn).T

        nk[i] = (
            np.r_[vis[1][1:][::-1], ir[1][1:][::-1]]
            + np.r_[vis[2][1:][::-1], ir[2][1:][::-1]] * 1j
        )

    w = 1e4 / np.r_[vis[0][1:][::-1], ir[0][1:][::-1]]
    ri = RefractiveIndices(w, nk)
    return Material("olivine 95/5", rho=3.3, ri=ri, **keywords)


def vacuum(wave=None, **keywords):
    """Vacuum.


    Parameters
    ----------
    wave : array, optional
        Use these wavelengths for creating the indices of refraction.

    """
    if wave is None:
        wave = np.logspace(-2, 3, 1000)
    ri = RefractiveIndices(wave, np.ones(len(wave), dtype=complex))
    return Material("vacuum", rho=0.0, ri=ri, **keywords)


def waterice(source="warren08", **keywords):
    """Water ice.

    Latent heat of sublimation from Delsemme & Miller 1971.  Vapor
    pressure from Lichtenegger and Komle 1991.

    Parameters
    ----------
    source : string
        * "warren08" for Warren and Brandt 2008
        * "warren84" for Warren 1984
        * "bertie69" for Bertie et al. 1969

    """

    ref = resources.files("grains2") / f"data/{source}.dat"
    with resources.as_file(ref) as fn:
        x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])

    def H(T):
        """Latent heat of sublimation (Delsemme & Miller 1971). [J/kg]"""
        return 2.888e6 - 1116.0 * T

    def Pv(T):
        """Vapor pressure (Lichtenegger & Komle 1991). [N/m2]"""
        Pr = 1e5  # N/m2
        Tr = 373.0  # K
        kB = 1.3806488e-23  # J/K
        mu = 18 * 1.66e-27  # molecular mass, kg
        return Pr * np.exp(mu * H(T) / kB * (1.0 / Tr - 1.0 / T))

    def sputtering(rh):
        """Rate of sputtering from solar wind for pure ice.

        Nominal solar wind, Mukai & Schwem 1981.


        Parameters
        ----------
        rh : float
            Heliocentric distance.  [au]

        Returns
        -------
        Z_sp : Quantity
            Sputtering rate per cross sectional area.

        """

        return u.Quantity(1.1e8 * rh**-2, "1/(s cm2)")

    return Ice(
        "water ice",
        rho=1.0,
        ri=ri,
        Pv=Pv,
        H=H,
        mu=18 * 1.66e-27,
        sputtering=sputtering,
        **keywords,
    )
