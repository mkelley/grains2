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

import os
import numpy as np
from pkg_resources import resource_filename

__all__ = [
    'amcarbon',
    'amolivine40',
    'amolivine50',
    'ampyroxene40',
    'ampyroxene50',
    'magnetite',
    'neutral',
    'olivine95',
    'vacuum',
    'waterice'
]


class RefractiveIndices(object):
    """Refractive index.

    Usage::
      ri = RefractiveIndices(wave, nk)

    Parameters
    ----------
    wave : ndarray
      The wavelengths. [micron]
    nk : complex ndarray, or dictionary
      The indices of refraction.  May be a dictionary of N-element
      arrays to specify different axes for anisotripic materials (see
      Examples).

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

    Methods
    --------
    callable self : returns `nk` at wavelength `w`, possibly through
      interpolation.

    copy : Return a copy of self.
    items : The names and indices of each axis, if available.
    keys : The names of each axis, if available.
    save : Save to a file.
    table : The indices as a table.
    values : The indices of each axis, if available.

    Examples
    --------
    Anisotripic materials can be specified, but attributes
    (``RefractiveIndices.n``) will return the mean of all axes.
    Individual axes can be returned by indexing the
    ``RefractiveIndices`` object:

      from grains import RefractiveIndices as RI
      w = np.linspace(0.3, 0.6)
      nk = np.ones((3, len(w)), complex)
      nk[1] *= 1.22
      nk[2] *= 1.5 + 0.01j
      ri = RI(w, dict(x=nk[0], y=nk[1], z=nk[2]))
      print(ri.n)
      print(ri['z'].n)

    """

    def __init__(self, wave, nk):
        self.wave = wave
        self.nk = nk

        if isinstance(self.nk, dict):
            # multiple indices were given
            self.axes = self.nk
            self.nk = np.zeros_like(self.wave, complex)
            for nk in self.axes.values():
                self.nk += nk
            self.nk /= len(self.axes)
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
          Set to True to interpolate `k` in logspace.

        Returns
        -------
        nk : array
          The interpolated values, or 0 if not defined at `wave`.

        """

        from scipy.interpolate import splrep, splev
        import astropy.units as u

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

        self.table().write(filename, format='ascii.fixed_width_two_line')

    def table(self):
        """The refractive indices as a table."""

        from astropy.table import Table, Column

        t = Table()
        t.add_column(Column(name='wave', data=self.wave))
        if len(self.axes) > 0:
            for k, v in self.axes:
                t.add_column(Column(name=k, data=v))
        else:
            t.add_column(Column(name='nk', data=self.nk))

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
    from astropy.io import ascii
    t = ascii.read(filename, format='fixed_width_two_line')
    if len(t.colnames) == 2:
        return RefractiveIndices(t['wave'].data, t['nk'].data)
    else:
        axes = dict()
        for k in t.colnames[1:]:
            axes[k] = t[k].data
        return RefractiveIndices(t['wave'].data, axes)


class Material(object):
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
      Latent heat of sublimation at temperature ``T`` in K: H(T).
      [J/kg]

    """

    def __init__(self, name, **keywords):
        self.name = name
        self.rho = keywords.get('rho')
        self.ri = keywords.get('ri')
        self.mu = keywords.get('mu')
        self.Pv = keywords.get('Pv', lambda T: None)
        self.H = keywords.get('H', lambda T: None)

    def __repr__(self):
        return "<grains.materials.Material [{}]>".format(self.name)


def amcarbon(**keywords):
    """Amorphous carbon."""
    fn = resource_filename('grains2', os.path.join('data', 'am-carbon.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous carbon", rho=1.5, ri=ri, **keywords)


def amolivine40(**keywords):
    """"Amorphous" olivine with Mg/Fe = 40/60."""
    fn = resource_filename('grains2', os.path.join('data', 'olivine-mg40.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous olivine 40/60", rho=3.3, ri=ri, **keywords)


def amolivine50(**keywords):
    """"Amorphous" olivine with Mg/Fe = 50/50."""
    fn = resource_filename('grains2', os.path.join('data', 'olivine-mg50.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous olivine 50/50", rho=3.3, ri=ri, **keywords)


def ampyroxene40(**keywords):
    """"Amorphous" pyroxene with Mg/Fe = 40/60."""
    fn = resource_filename(
        'grains2', os.path.join('data', 'pyroxene-mg40.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous pyroxene 40/60", rho=3.3, ri=ri, **keywords)


def ampyroxene50(**keywords):
    """"Amorphous" pyroxene with Mg/Fe = 50/50."""
    fn = resource_filename(
        'grains2', os.path.join('data', 'pyroxene-mg50.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])
    return Material("amorphous pyroxene 50/50", rho=3.3, ri=ri, **keywords)


def magnetite(**keywords):
    """Magnetite, Fe3O4."""
    fn = resource_filename('grains2', os.path.join('data', 'magnetite.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[:, 0], x[:, 1] + 1j * x[:, 2])
    return Material("magnetite", ri=ri, **keywords)


def wustite(**keywords):
    """Wüstite, Fe3O4."""
    fn = resource_filename('grains2', os.path.join('data', 'fe100o.txt'))
    x = np.loadtxt(fn, skiprows=4).T
    ri = RefractiveIndices(x[:, 0], x[:, 1] + 1j * x[:, 2])
    return Material("Wüstite", ri=ri, **keywords)


def ferropericlase50(**keywords):
    """Ferropericlase Mg_0.5 Fe_0.5 O."""
    fn = resource_filename('grains2', os.path.join('data', 'fe50o.txt'))
    x = np.loadtxt(fn, skiprows=4).T
    ri = RefractiveIndices(x[:, 0], x[:, 1] + 1j * x[:, 2])
    return Material("ferropericlase 50", ri=ri, **keywords)


def neutral(nk, wave=None, **keywords):
    """A neutral scatterer, specify nk.

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
    fn = resource_filename('grains2', os.path.join('data', 'oliv_vis.txt'))
    vis = np.loadtxt(fn).T

    nk = dict()
    for i in 'xyz':
        fn = resource_filename(
            'grains2', os.path.join('data', 'oliv_nk_{}.txt'.format(i)))
        ir = np.loadtxt(fn).T
        nk[i] = (np.r_[vis[1][1:][::-1], ir[1][1:][::-1]] +
                 np.r_[vis[2][1:][::-1], ir[2][1:][::-1]] * 1j)

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
    ri = RefractiveIndices(wave, np.ones(len(wave), dtype=np.complex))
    return Material("vacuum", rho=0.0, ri=ri, **keywords)


def waterice(source='warren08', **keywords):
    """Water ice.

    Latent heat of sublimation from Delsemme & Miller 1971.  Vapor
    pressure from Lichtenegger and Komle 1991.

    Parameters
    ----------
    source : string
      warren08 for Warren and Brandt 2008
      warren84 for Warren 1984
      bertie69 for Bertie et al. 1969

    """

    fn = resource_filename('grains2', os.path.join('data', source + '.dat'))
    x = np.loadtxt(fn).T
    ri = RefractiveIndices(x[0], x[1] + 1j * x[2])

    def H(T):
        """Latent heat of sublimation (Delsemme & Miller 1971). [J/kg]"""
        return 2.888e6 - 1116. * T

    def Pv(T):
        """Vapor pressure (Lichtenegger & Komle 1991). [N/m2]"""
        Pr = 1e5  # N/m2
        Tr = 373.0  # K
        kB = 1.3806488e-23  # J/K
        mu = 18 * 1.66e-27  # molecular mass, kg
        return Pr * np.exp(mu * H(T) / kB * (1.0 / Tr - 1.0 / T))

    return Material("water ice", rho=1.0, ri=ri, Pv=Pv, H=H,
                    mu=18 * 1.66e-27, **keywords)
