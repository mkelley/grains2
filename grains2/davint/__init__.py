_davint_err = dict()
_davint_err[2] = 'x1 was less than x0'
_davint_err[3] = 'the number of x between x0 and x1 (inclusive) was less than 3 and neither of the two special cases described in the abstract occurred.  No integration was performed.'
_davint_err[4] = 'the restriction x(i+1) > x(i) was violated.'
_davint_err[5] = 'the number of function values was < 2'

from .davint import davint as _davint

def davint(x, y, x0, x1, axis=0):
    """Integrate an array using overlapping parabolas.
    
    Interface to davint.f from SLATEC at netlib.org.

    DAVINT integrates a function tabulated at arbitrarily spaced
    abscissas.  The limits of integration need not coincide with the
    tabulated abscissas.

    A method of overlapping parabolas fitted to the data is used
    provided that there are at least 3 abscissas between the limits of
    integration.  DAVINT also handles two special cases.  If the
    limits of integration are equal, DAVINT returns a result of zero
    regardless of the number of tabulated values.  If there are only
    two function values, DAVINT uses the trapezoid rule.

    Parameters
    ----------
    x : ndarray
      Abscissas, must be in increasing order.
    y : ndarray
      Function values.
    x0 : float
      Lower limit of integration.
    x1 : float
      Upper limit of integration.
    axis : int
      If `y` is a 2D array, then integrate over axis `axis` for each
      element of the other axis.

    Returns
    -------
    float
      The result.

    """

    import numpy as np

    y = np.array(y)
    if y.ndim == 1:
        r, ierr = _davint(x, y, len(x), x0, x1)
        if ierr != 1:
            raise RuntimeError("DAVINT integration error: {}".format(
                _davint_err[ierr]))
    elif y.ndim == 2:
        r = np.zeros(y.shape[axis])
        for i, yy in enumerate(np.rollaxis(y, axis)):
            r[i] = davint(x, yy, x0, x1)
    else:
        raise ValueError("y must have 1 or 2 dimensions.")

    return r
