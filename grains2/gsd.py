"""
gsd --- Grain size distributions
================================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   GSD

   Hanner
   Hansen
   Normal
   PowerLaw

   Functions
   ---------
   hanner_gsd
   hansen_gsd
   powerlaw_gsd

"""

import numpy as np

__all__ = [
    'Hanner',
    'Hansen',
    'Normal',
    'PowerLaw',
    'hanner_gsd',
    'hansen_gsd',
    'powerlaw_gsd'
]

class GSD(object):
    """Abstract base class for grain size distributions."""

    def n(self, a):
        """Evaulate the distribution.

        Parameters
        ----------
        a : float or array
          Grain radius.

        Returns
        -------
        float or array

        """
        return self._n(a)

class Hanner(GSD):
    """Hanner modified power law.

    Parameters
    ----------
    a0 : float
      Minimum grain radius.  Same units as `a`.
    N : float
      GSD for large grains (`a >> ap`) is `a**-N`.
    M : float, optional
      `ap = a0 * (M + N) / N`.  One of `M` or `ap` must be provided.
    ap : float, optional
      Peak grain radius.  One of `M` or `ap` must be provided.  Same
      units as `a`.
    Np : float, optional
      Number of grains with radius `ap`.

    """

    def __init__(self, a0, N, M=None, ap=None, Np=1.0):
        test = (M is not None) or (ap is not None)
        assert test, "One of M or ap must be provided."

        test = (((M is None) and (ap is not None))
                or ((M is not None) and (ap is None)))
        assert test, "Only one of M or ap can be provided."

        self.a0 = a0
        self.N = N
        self.Np = Np

        if M is None:
            self.M = (ap / a0 - 1) * N
        else:
            self.M = M

    @property
    def ap(self):
        return self.a0 * (self.M + self.N) / self.N

    @ap.setter
    def ap(self, ap_):
        self.M = (ap_ / self.a0 - 1) * self.N

    def _n(self, a):
        norm = (1 - self.a0 / self.ap)**self.M * (self.a0 / self.ap)**self.N
        n = (1 - self.a0 / a)**self.M * (self.a0 / a)**self.N
        return n / norm

class Hansen(GSD):
    """Hansen modified gamma distribution.

    Used extensively in Hansen and Travis (1974).

    n(a) = constant a**((1 - 3 a_var) / a_var) exp(-a / (a_eff a_var))
      a_eff = mean effective radius
      a_var = relative effective radius variance, 0 <= a_var < 0.5

    The mean effective radius is the mean radius weighed by particle
    area.

    The effective skewness is `2 * sqrt(a_var)`.  For small `a_var`,
    the distribution is strongly peaked at `a_eff`.

    Hansen 1971, J. Atmospheric Sci. 28, 1400.

    Parameters
    ----------
    a_eff : float
      Mean effective grain radius.  Same units as `a`.
    a_var : float
      Relative effective radius variance.
    N : float, optional
      Total number of grains.

    Attributes
    ----------
    C : float
      The normalization constant.

    """

    def __init__(self, a_eff, a_var, N=1.0):
        test = (0 <= a_var) * (a_var < 0.5)
        assert test, "0 <= a_var < 0.5 not satisfied."

        self.a_eff = a_eff
        self.a_var = a_var
        self.N = N

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, n):
        from scipy.special import gamma
        self._N = n
        if self.a_var == 0.0:
            self.C = 1.0
        else:
            x = ((2 * self.a_var - 1) / self.a_var)
            self.C = self._N
            self.C *= (self.a_eff * self.a_var)**x
            self.C /= gamma((1 - 2 * self.a_var) / self.a_var)

    def _n(self, a):
        if self.a_var == 0.0:
            return self.C
        else:
            x = np.exp(-a / (self.a_eff * self.a_var))
            if np.size(x) == 1:
                if x == 0.0:
                    n = 0.0
                else:
                    n = self.C * a**((1 - 3 * self.a_var) / self.a_var) * x
            else:
                i = (x == 0.0)
                n = np.zeros_like(x)
                if any(i):
                    x[i] = 0.0
                n[~i] = (self.C
                         * a[~i]**((1 - 3 * self.a_var) / self.a_var)
                         * x[~i])
            return n

class Normal(GSD):
    """Normal distribution.

    Parameters
    ----------
    mu : float
      Mean grain radius.
    sigma : float
      Variance about the mean.
    Np : float, optional
      Number of grains with radius `mu`.  Only one of `Np` or `N` can
      be specified.
    N : float, optional
      Total number of grains.  Only one of `Np` or `N` can be
      specified.

    """

    def __init__(self, mu, sigma, Np=None, N=1.0):
        assert sigma > 0, "sigma <= 0"
        assert mu > 0, "mu <= 0"

        test = (Np is None) != (N is None)
        assert test, "Only one of N or Np may be provided."

        self.mu = mu
        self.sigma = sigma
        if Np is None:
            self.N = N
        else:
            self.Np = Np

    def _integral(self):
        from scipy.special import erf
        return 0.5 * (1 + erf(self.mu / self.sigma / np.sqrt(2.0)))

    @property
    def N(self):
        return self.Np * self.sigma * np.sqrt(2 * np.pi) * self._integral()

    @N.setter
    def N(self, n):
        self.Np = n / self.sigma / np.sqrt(2 * np.pi) / self._integral()

    def _n(self, a):
        return self.Np * np.exp(-(a - self.mu)**2 / (2 * self.sigma**2))

class PowerLaw(GSD):
    """Power law grain size distribution.

    Parameters
    ----------
    N : float
      Power-law slope.
    a0 : float, optional
      Normalization grain radius.
    N0 : float, optional
      Number of grains with radius `a0`.

    """

    def __init__(self, N, a0=1.0, N0=1.0):
        self.a0 = a0
        self.N = N
        self.N0 = N0

    def _n(self, a):
        return self.N0 * (a / self.a0)**self.N

def hanner_gsd(a, a0, N, M=None, ap=None, Np=1.0):
    """Hanner modified power law grain size distribuion.

    n(a) = Np * (1 - a0 / a)**M * (a0 / a)**N

    Parameters
    ----------
    a : float or array
      Grain radius.
    a0 : float
      Minimum grain radius.  Same units as `a`.
    N : float
      GSD for large grains (`a >> ap`) is `a**-N`.
    M : float, optional
      `ap = a0 * (M + N) / N`.  One of `M` or `ap` must be provided.
    ap : float, optional
      Peak grain radius.  One of `M` or `ap` must be provided.  Same
      units as `a`
    Np : float, optional
      Number of grains with radius `ap`.

    Returns
    -------
    float or array

    """

    test = (M is not None) or (ap is not None)
    assert test, "One of M or ap must be provided."

    test = (((M is None) and (ap is not None))
            or ((M is not None) and (ap is None)))
    assert test, "Only one of M or ap can be provided."

    if M is None:
        M = (ap / a0 - 1) * N

    return Np * (1 - a0 / a)**M * (a0 / a)**N

def hansen_gsd(a, a_eff, a_var, N=1.0):
    """Hansen modified gamma distribution.

    Used extensively in Hansen and Travis (1974).

    n(a) = constant a**((1 - 3 a_var) / a_var) exp(-a / (a_eff a_var))
      a_eff = effective radius
      a_var = relative effective radius variance, 0 <= a_var < 0.5

    The mean effective radius is the mean radius weighed by particle
    area.

    The effective skewness is `2 * sqrt(a_var)`.  For small `a_var`,
    the distribution is strongly peaked at `a_eff`.

    Hansen 1971, J. Atmospheric Sci. 28, 1400.

    Parameters
    ----------
    a : float or array
      Grain radius.
    a_eff : float
      Mean effective grain radius.  Same units as `a`.
    a_var : float
      Relative effective radius variance.
    N : float, optional
      Total number of grains.

    """

    from scipy.special import gamma

    test = (0 <= a_var) * (a_var < 0.5)
    assert test, "0 <= a_var < 0.5 not satisfied."

    C = (N * (a_eff * a_var)**((2 * a_var - 1) / a_var)
         / gamma((1 - 2 * a_var) / a_var))

    return C * a**((1 - 3 * a_var) / a_var) * np.exp(-a / (a_eff * a_var))

def powerlaw_gsd(a, N, a0=1.0, N0=1.0):
    """Power law grain size distribuion.

    n(a) = N0 * (a/a0)**N

    Parameters
    ----------
    a : float or array
      Grain radius.
    N : float
      Power-law slope.
    a0 : float
      GSD normalization radius.  Same units as `a`.
    N0 : float, optional
      Number of grains with radius `a0`.

    Returns
    -------
    float

    """
    return N0 * (a / a0)**N

# update module docstring
from mskpy.util import autodoc
autodoc(globals())
del autodoc
