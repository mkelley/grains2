"""
porosity --- Porosity models
============================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   PorosityModel

   ConstantPorosity
   FractalPorosity
   Solid

   Functions
   ---------
   fractal_porosity

"""

import numpy as np

__all__ = [
    'ConstantPorosity',
    'FractalPorosity',
    'Solid',
    'fractal_porosity'
]

class PorosityModel(object):
    """Abstract base class for porosity models.

    Parameters
    ----------
    emt : EMT, optional
      The effective medium theory to use for mixing vacuum into the
      grain.  The default is Bruggeman EMT.

    """

    def __init__(self, emt=None):
        from .emt import EMT, Bruggeman

        self.emt = Bruggeman() if emt is None else emt
        assert isinstance(self.emt, EMT)

    def __call__(self, material, a):
        """Mix vacuum into a material to make it porous.

        Parameters
        ----------
        material : Material
          The material to make porous.
        a : float
          Grain radius.

        Returns
        -------
        Material

        """
        return self._mix(material, a)

class ConstantPorosity(PorosityModel):
    """Porosity independent of radius.

    Parameters
    ----------
    p : float
      The volume fraction of vaccum, i.e., porosity.
    emt : EMT, optional
      The effective medium theory to use for mixing vacuum into the
      grain.  The default is Bruggeman EMT.

    """

    def __init__(self, p, emt=None):
        PorosityModel.__init__(self, emt=emt)
        self.p = p
        assert self.p >= 0
        assert self.p <= 1.0

    def _mix(self, material, a):
        from .material import vacuum
        return self.emt.mix((material, vacuum(wave=material.ri.wave)),
                            (1 - self.p, self.p))

class FractalPorosity(PorosityModel):
    """Porosity for a fractally structured grain.

    Parameters
    ----------
    a0 : float
      Basic unit radius.  Grains smaller than `a0` are always solid.
    D : float
      Fractal dimension.  Solid grains have `D = 3`.
    emt : EMT, optional
      The effective medium theory to use for mixing vacuum into the
      grain.  The default is Bruggeman EMT.

    """

    def __init__(self, a0, D, emt=None):
        PorosityModel.__init__(self, emt=emt)
        self.a0 = a0
        assert self.a0 > 0
        self.D = D
        assert self.D <= 3.0

    def _mix(self, material, a):
        from .material import vacuum
        p = fractal_porosity(a, self.a0, self.D)
        return self.emt.mix((material, vacuum(wave=material.ri.wave)),
                            (1 - p, p))

class Solid(ConstantPorosity):
    """Solid grains."""

    def __init__(self):
        ConstantPorosity.__init__(self, 0.0)

def fractal_porosity(a, a0, D):
    """Porosity for a fractally structured grain.

    Parameters
    ----------
    a : float or ndarray
      Grain radius.
    a0 : float
      Basic unit radius.  Grains smaller than `a0` are always solid.
      Same units as `a`.
    D : float
      Fractal dimension.

    Returns
    -------
    P : float or ndarray
      The porosity of the grain(s).  P = 1.0 is 100% porous (i.e.,
      vacuum).

    """
    P = 1.0 - (a / a0)**(D - 3.0)
    if np.array(a).size == 1:
        if P < 0:
            P = 0.0
    else:
        if np.any(P < 0):
            P[P < 0] = 0.0
    return P

# update module docstring
from mskpy.util import autodoc
autodoc(globals())
del autodoc
