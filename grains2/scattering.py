"""
scattering --- Scattering models
================================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   ScatteringModel
   Mie
   OblateCDE
   ProlateCDE

"""

from abc import ABC, abstractmethod
from functools import wraps
import numpy as np
from numpy import pi
from .bhmie import bhmie
from .material import RefractiveIndices

__all__ = ["Mie", "OblateCDE", "ProlateCDE"]


def _return_float_or_array(func):
    """Function decorator that returns len 1 arrays as scalars."""

    @wraps(func)
    def wrapper(*args, **keywords):
        r = func(*args, **keywords)
        if len(r) == 1:
            return r[0]
        else:
            return r

    return wrapper


def _not_implemented(func):

    def wrapper(*args, **keywords):
        raise NotImplementedError(f"{func.__name__} is not implemented for this model.")

    return wrapper


class ScatteringModel(ABC):
    """Base class for light scattering.

    Several functions may be defined, but none are specifically required in
    order to support models which cannot calculate some values.


    """

    def __init__(self):
        pass

    @_not_implemented
    def q(self, a, w, ri):
        """Scattering properties.


        Parameters
        ----------
        a : float or array
            Grain size.  [micron]

        w : float or array
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or array
            Refractive index of the material.

        Returns
        -------
        q : dictionary
            All scattering properties as a dictionary: qsca, qext, qabs, qback,
            gsca, s1, s2, Phi.

        """
        pass

    @_not_implemented
    def qsca(self, a, w, ri):
        """Light scattering efficiency.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or array
            Refractive index of the material.


        Returns
        -------
        qsca : float
            The scattering efficiency.

        """
        pass

    @_not_implemented
    def qext(self, a, w, ri):
        """Extinction efficiency.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.


        Returns
        -------
        qext : float
            The extinction efficiency.

        """
        pass

    @_not_implemented
    def qabs(self, a, w, ri):
        """Absorption efficiency.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.


        Returns
        -------
        qabs : float
            The absorption efficiency.

        """
        pass

    @_not_implemented
    def qback(self, a, w, ri):
        """Back scattering efficiency.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.


        Returns
        -------
        qback : float
            Back scattering efficiency.

        """
        pass

    @_not_implemented
    def gsca(self, a, w, ri):
        """Mean scattering angle (<cos(th)>).


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.


        Returns
        -------
        gsca : float
            <cos(th)>.

        """
        pass

    @_not_implemented
    def s1(self, a, w, ri):
        """Scattering of light perpendicular to the scattering plane.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.


        Returns
        -------
        s1 : ndarray
            Diagonal elements of the scattering matrix for E perpendicular
            incident light.

        th : ndarray
            Angles for ``s1``.

        """
        pass

    @_not_implemented
    def s2(self, a, w, ri):
        """Scattering of light parallel to the scattering plane.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.


        Returns
        -------
        s2 : ndarray
            Diagonal elements of the scattering matrix for E parallel incident
            light.

        th : ndarray
            Angles for ``s2``.

        """
        pass

    @_not_implemented
    def Phi(self, a, w, qsca, s1, s2):
        """Phase function.

        Angles are a function of the scattering angle ``alpha``.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        qsca : float or ndarray
            Scattering efficiency.

        s1, s2 : float or ndarray
            Scattering elements for E perp. and E para.  Set to `0j` to disable
            that component.


        Returns
        -------
        Phi : ndarray
            Phase function.

        """

        k = 2 * pi / w
        Csca = qsca * pi * a**2
        s11 = (np.abs(s2) ** 2 + np.abs(s1) ** 2) / 2
        Phi = s11 / k**2 / Csca

        return Phi

    @staticmethod
    def _process_input(a, w, ri):
        """Test inputs, and prepare x, nk, and q variables.

        1) Compute x = 2 pi a / w:

            If a is an array of len N, and w is of len M, return NxM x for each
            combination of a and w.

            Otherwise, return an x with shape max(len(a), len(x))

        2) Compute nk:

            If ri is a RefractiveIndices instance, it will be interpolated onto
            w.

            If ri is a single value, it will be repeated for all x.

            If ri is an array of length N, it is assumed each element
            corresponds to each w.


        Returns
        -------
        x : ndarray
            Size parameters (does not account for imaginary index).

        nk : ndarray
            Refractive indices.

        q : ndarray
            q will have the same number of elements as x.

        shape : tuple
            q will need to be reshaped to this shape in order to follow the
            above rules.

        """

        a = np.array(a)
        w = np.array(w)
        if (a.size > 1) and (w.size > 1):
            x = 2 * pi * np.outer(a, 1.0 / w)
        else:
            x = 2 * pi * a / w

        if not np.iterable(x):
            x = np.array([x])
        shape = x.shape
        x = x.flatten()

        q = np.zeros_like(x)

        if isinstance(ri, RefractiveIndices):
            nk = ri(w)
            if len(shape) == 2:
                nk = np.tile(nk, shape[0])
            return x, nk, q, shape
        if np.size(ri) == 1:
            # assume it is already at the correct wavelength
            return x, np.ones_like(x) * ri, q, shape
        if np.size(ri) == np.size(w):
            # assume it is already at the correct wavelengths
            nk = ri
            if len(shape) == 2:
                nk = np.tile(nk, shape[0])
            return x, nk, q, shape
        raise ValueError(
            "ri should be a RefractiveIndices instance,"
            " a single value, or a list of values"
            " at each w."
        )


class Mie(ScatteringModel):
    """Mie scattering.

    Uses Bruce Drain's bhmie.f.

    """

    def q(self, a, w, ri, nang=90):
        """Scattering properties.


        Parameters
        ----------
        a : float or array
            Grain size.  [micron]

        w : float or array
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or array
            Refractive index of the material.

        nang : int
            Number of angles to compute for s1, s2 between 0 and 90 deg.
            ``bhmie`` will calculate ``2 * nang - 1`` directions from 0 to 180
            deg.  For example, ``nang=2`` yields ``theta=[0, 90, 180]``.


        Returns
        -------
        q : dictionary
            All scattering properties as a dictionary: qsca, qext, qabs, qback,
            gsca, s1, s2, Phi.  Also, `alpha`, the scattering angles for s1, s2,
            etc.

        """

        nang = 2 if nang < 2 else nang
        x, nk, qsca, shape = self._process_input(a, w, ri)
        qext, qback, gsca = np.ones((3,) + qsca.shape)
        s1, s2 = np.zeros((2, qsca.size, 2 * nang - 1), complex)
        # Phi = np.zeros((qsca.size, 2 * nang - 1))
        alpha = np.linspace(0, 180.0, 2 * nang - 1)
        for i in range(len(x)):
            q = bhmie(x[i], nk[i], nang)
            qsca[i] = q[3]
            qext[i] = q[2]
            qback[i] = q[4]
            gsca[i] = q[5]
            s1[i] = q[0][: s1.shape[1]]
            s2[i] = q[1][: s1.shape[1]]
            # Phi[i] = self.Phi(a[i], w[i], qsca[i], s1[i], s2[i])

        return dict(
            qsca=qsca,
            qext=qext,
            qabs=qext - qsca,
            qback=qback,
            gsca=gsca,
            s1=s1,
            s2=s2,  # Phi=Phi,
            alpha=alpha,
        )

    @_return_float_or_array
    def qsca(self, a, w, ri):
        x, nk, q, shape = self._process_input(a, w, ri)
        for i in range(len(x)):
            q[i] = bhmie(x[i], nk[i], 1)[3]
        return q.reshape(shape)

    @_return_float_or_array
    def qext(self, a, w, ri):
        x, nk, q, shape = self._process_input(a, w, ri)
        for i in range(len(x)):
            q[i] = bhmie(x[i], nk[i], 1)[2]
        return q.reshape(shape)

    @_return_float_or_array
    def qabs(self, a, w, ri):
        x, nk, q, shape = self._process_input(a, w, ri)
        for i in range(len(x)):
            r = bhmie(x[i], nk[i], 1)
            q[i] = r[2] - r[3]
        return q.reshape(shape)

    @_return_float_or_array
    def qback(self, a, w, ri):
        x, nk, q, shape = self._process_input(a, w, ri)
        for i in range(len(x)):
            q[i] = bhmie(x[i], nk[i], 1)[4]
        return q.reshape(shape)

    @_return_float_or_array
    def gsca(self, a, w, ri):
        x, nk, g, shape = self._process_input(a, w, ri)
        for i in range(len(x)):
            g[i] = bhmie(x[i], nk[i], 1)[5]
        return g.reshape(shape)

    def s1(self, a, w, ri, nang=23):
        """Scattering matrix for light perpendicular to the scattering plane.


        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.

        nang : int
            Number of angles to compute, from 0 to 90 degrees.


        Returns
        -------
        s1 : ndarray
            Diagonals of the scattering matrix, each has length of ``nang * 2 -
            1``.

        """
        x, nk = self._process_input(a, w, ri)[:2]
        nang = 2 if nang < 2 else nang
        s = np.zeros((len(x), 2 * nang - 1))
        for i in range(len(x)):
            s[i] = bhmie(x[i], nk[i], nang)[0][: s.shape[1]]
        return s

    def s2(self, a, w, ri, nang=23):
        """Scattering matrix for light parallel to the scattering plane.

        Parameters
        ----------
        a : float or ndarray
            Grain size.  [micron]

        w : float or ndarray
            Wavelength of scattered light.  [micron]

        ri : RefractiveIndices, complex, or ndarray
            Refractive index of the material.

        nang : int
            Number of angles to compute, from 0 to 90 degrees.


        Returns
        -------
        s2 : ndarray
            Diagonals of the scattering matrix, each has length of ``nang * 2 -
            1``.

        """

        x, nk = self._process_input(a, w, ri)[:2]
        nang = 2 if nang < 2 else nang
        s = np.zeros((len(x), 2 * nang - 1))
        for i in range(len(x)):
            s[i] = bhmie(x[i], nk[i], nang)[1][: s.shape[1]]
        return s


class OblateCDE(ScatteringModel):
    """CDE oblate ellipsoids aligned along crystallographic axes.

    Refractive indices should be specified for three axes.


    Parameters
    ----------
    axis : string
        The name of the elongated axis.

    factor : float
        The relative scale factor of the elongation along ``axis``.  The other
        axes will have a scale factor of 1.  These factors are used as a, b, and
        c, in the above equations.


    Attributes
    ----------
    axis : float, read-only

    factor : float, read-only

    L : ndarray, read-only

    e : float, read-only


    Notes
    -----

    A continuous distribution of a ramdomly oriented collection of identical
    homogeneous ellipsoids (Borhen and Huffman 1983, p354):

        <Cabs> = k * v / 3 * imag(sum(1 / (beta + L[i])))

        beta = 1 / (nk**2 - 1)

        v = volume = 4 / 3 * pi * a * b * c

    ``a``, ``b``, and ``c`` are the semi-major axes of the ellipsoid.

    Oblates (Borhen and Huffman 1983, p 146):

        L[0] = g(e) / 2.0 / e**2 * (pi / 2.0 - arctan(g(e))) - g(e)**2 / 2.0

        L[1] = L[0] L[2] = (1.0 - L[0] * 2) g(e) = sqrt((1.0 - e**2) / e**2)

        e**2 = 1.0 - c**2 / a**2

    ``c`` is the shortened axis.

    """

    def __init__(self, axis, factor):
        assert factor <= 1, "Oblates must have factor <= 1"

        self._axis = axis
        self._factor = factor

        self._L = np.zeros(3)
        if factor == 1:
            self._L += 1.0 / 3.0
            self._e = 0.0
        else:
            e2 = 1.0 - self._factor**2
            e = np.sqrt(e2)
            g2 = (1.0 - e2) / e2
            g = np.sqrt(g2)
            self._e = e
            self._L[0] = g / 2.0 / e2 * (pi / 2.0 - np.arctan(g)) - g2 / 2.0
            self._L[1] = self._L[0]
            self._L[2] = 1.0 - self._L[0] * 2

    @property
    def axis(self):
        """The crystallographic axis aligned with the shortened axis."""
        return self._axis

    @property
    def factor(self):
        """The relative size of the shortened axis."""
        return self._factor

    @property
    def L(self):
        """The geometrical scaling factors."""
        return self._L

    @property
    def e(self):
        """The eccentricity of the ellipsoid."""
        return self._e

    def qabs(self, av, w, ri):
        nk = dict()
        for axis in ri.keys():
            x, nk[axis], q, shape = self._process_input(av, w, ri[axis])

        sigma = np.zeros_like(x, dtype=complex)
        for axis in ri.keys():
            beta = 1.0 / (nk[axis] ** 2 - 1.0)
            if axis == self.axis:
                sigma += 1.0 / (beta + self.L[2])
            else:
                sigma += 1.0 / (beta + self.L[0])

        q = x * 4.0 / 9.0 * np.imag(sigma)
        return q.reshape(shape)


class ProlateCDE(ScatteringModel):
    """CDE prolate ellipsoids aligned along crystallographic axes.

    Refractive indices should be specified for three axes.


    Parameters
    ----------
    axis : string
        The name of the elongated axis.

    factor : float
        The relative scale factor of the elongation along ``axis``.  The other
        axes will have a scale factor of 1.  These factors are used as a, b, and
        c, in the above equations.


    Attributes
    ----------
    axis : float, read-only

    factor : float, read-only

    L : ndarray, read-only

    e : float, read-


    Notes
    -----

    A continuous distribution of a ramdomly oriented collection of identical
    homogeneous ellipsoids (Borhen and Huffman 1983, p354):

        <Cabs> = k * v / 3 * imag(sum(1 / (beta + L[i])))

        beta = 1 / (nk**2 - 1)

        v = volume = 4 / 3 * pi * a * b * c

    ``a``, ``b``, and ``c`` are the semi-major axes of the ellipsoid.

    Prolates (Borhen and Huffman 1983, p146):

        L[0] = (1.0 - e**2) / e**2 * (-1.0 + 1.0 / 2.0 / e *
                                      log((1 + e) / (1.0 - e)))

        L[1] = L[2] = (1.0 - L[0]) / 2.0

        e**2 = 1.0 - b**2 / a**2

    ``a`` is the elongated axis.

    """

    def __init__(self, axis, factor):
        assert factor >= 1, "Prolates must have factor >= 1"

        self._axis = axis
        self._factor = factor

        self._L = np.zeros(3)
        if factor == 1:
            self._L += 1.0 / 3.0
            self._e = 0.0
        else:
            e2 = 1.0 - 1.0 / self._factor**2
            e = np.sqrt(e2)
            self._e = e
            self._L[0] = (
                (1.0 - e2) / e2 * (np.log((1.0 + e) / (1.0 - e)) / 2.0 / e - 1.0)
            )
            self._L[1] = (1.0 - self._L[0]) / 2.0
            self._L[2] = self._L[1]

    @property
    def axis(self):
        """The crystallographic axis aligned with the elongated axis."""
        return self._axis

    @property
    def factor(self):
        """The relative size of the elonated axis."""
        return self._factor

    @property
    def L(self):
        """The geometrical scaling factors."""
        return self._L

    @property
    def e(self):
        """The eccentricity of the ellipsoid."""
        return self._e

    def qabs(self, av, w, ri):
        # k = x / av
        # v = 4.0 / 3.0 * pi * av**3
        # Cabs = k * v / 3.0 * np.imag(sigma)
        # q = Cabs / pi / av**2
        # q = k * v / 3.0 * np.imag(sigma) / pi / av**2
        # q = x / av * 4.0 / 3.0 * pi * av**3 / 3.0 * np.imag(sigma) / pi / av**2

        nk = dict()
        for axis in ri.keys():
            x, nk[axis], q, shape = self._process_input(av, w, ri[axis])

        sigma = np.zeros_like(x, dtype=complex)
        for axis in ri.keys():
            beta = 1.0 / (nk[axis] ** 2 - 1.0)
            if axis == self.axis:
                sigma += 1.0 / (beta + self.L[0])
            else:
                sigma += 1.0 / (beta + self.L[1])

        q = x * 4.0 / 9.0 * np.imag(sigma)
        return q.reshape(shape)


# update module docstring
from mskpy.util import autodoc

autodoc(globals())
del autodoc
