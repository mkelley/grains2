"""
emt --- Effective medium theory
===============================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   EMT
   Bruggeman

"""

import numpy as np

__all__ = [
    'Bruggeman'
]

class EMT(object):
    """Abstract base class for effective medium theory classes."""

    def __init__(self):
        pass

    def mix(self, materials, fractions, name=None):
        """Mix materials together.

        Parameters
        ----------
        materials : array of Materials
          The materials to mix.
        fractions : array
          The relative fractions of each material (does not need to be
          normalized to 1.0).
        name : string, optional
          A new name for the material.  If `None`, a name will be
          autogenerated from the inputs.

        Returns
        -------
        mix : Material
          The mixture of `materials`.

        """

        return self._mix(materials, fractions, name=name)

class Bruggeman(EMT):
    """Effective medium theory via the Bruggeman mixing rule.

    Bruggeman inclusions are randomly embedded in a matrix, and there
    is no distinction between the matrix and the inclusions in the
    formuale.

    The refractive indices of the embedded material are interpolated
    onto the wavelengths of the matrix.  Parts of this code is
    translated from David Harker's (UCSD) IDL routine mixture.pro,
    dated 11 May 1998.

    The materials are interpolated onto the intersection of the
    wavelength ranges.

    """

    def _mix(self, materials, fractions, name=None):
        from .material import Material, RefractiveIndices

        f = np.array(fractions, float) / np.sum(fractions)

        if (len(materials) != 2) or (len(f) != 2):
            raise ValueError("Only 2 materials (and fractions) are currently"
                             " supported.")

        for m in materials:
            assert isinstance(m, Material)

        if name is None:
            name = '{}({:.0%})+{}({:.0%})'.format(materials[0].name, f[0],
                                                  materials[1].name, f[1])

        wave = np.concatenate([m.ri.wave for m in materials])
        wave = np.unique(wave)
        ep = np.vstack([m.ri(wave) for m in materials])**2
        ep[~np.isfinite(ep)] = 0.0
        i = np.prod(np.real(ep), 0) == 0
        if any(i):
            wave = wave[~i]
            ep = ep[:, ~i]

        b = f[0] * (2.0 * ep[0] - ep[1]) + f[1] * (2.0 * ep[1] - ep[0])
        mixture = np.sqrt(0.25 * (b + np.sqrt(b**2 + 8.0 * ep[0] * ep[1])))
        rho = np.sum(np.array([m.rho for m in materials]) * f)

        return Material(name, rho=rho, ri=RefractiveIndices(wave=wave, nk=mixture))

# update module docstring
from mskpy.util import autodoc
autodoc(globals())
del autodoc