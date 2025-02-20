Grain porosity models
=====================

``grains2`` provides support for grain porosity models.  Porosity may be a function of grain radius.  A commonly used model is the fractal porosity, described by the fractal dimension :math:`D`, and the basic unit radius, :math:`a_0`:

.. math::

    P = 1.0 - (a / a_0)^{D - 3}

Note that :math:`D=3` is solid for all :math:`a`.

Calculate the porosity of a 1.0 μm grain, given :math:`D=2.8` and :math:`a_0=0.1` μm:

    >>> from grains2 import FractalPorosity
    >>>
    >>> a0 = 0.1
    >>> D = 2.8
    >>>
    >>> porosity = FractalPorosity(a0, D)
    >>> porosity.P(1.0)  # doctest: +FLOAT_CMP
    0.369042655519807

Plot the same porosity model for grains of size 0.1 μm to 1 mm:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from grains2 import FractalPorosity

    porosity = FractalPorosity(0.1, 2.8)
    a = np.logspace(-1, 3, 1000)

    fig, ax = plt.subplots()
    ax.plot(a, porosity.P(a), label="$D=2.8$")
    ax.legend()
    plt.setp(
        ax,
        xlabel="Radius (μm)",
        xscale="log",
        xlim=(0.1, 1000),
        ylabel="$P$",
        yscale="log",
    )
    plt.tight_layout()
