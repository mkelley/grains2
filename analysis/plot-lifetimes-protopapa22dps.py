import os

# from glob import glob
import numpy as np

# import scipy.ndimage as nd
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.io import ascii, fits

# from astropy.table import Table
# from astropy.time import Time
# from astropy.coordinates import SkyCoord
# from sbpy.data import Ephem
# from sbpy.activity import Afrho, Efrho
# import sbpy.units as sbu
import mskpy

pure = fits.getdata("lifetimes-waterice-protopapa22dps.fits")
a = fits.getdata("lifetimes-waterice-protopapa22dps.fits", extname="radius")
rh = fits.getdata("lifetimes-waterice-protopapa22dps.fits", extname="rh")
ac002 = fits.getdata(
    "lifetimes-waterice_99.8_+amorphouscarbon_0.2_-protopapa22dps.fits"
)
ac005 = fits.getdata(
    "lifetimes-waterice_99.5_+amorphouscarbon_0.5_-protopapa22dps.fits"
)
ac010 = fits.getdata(
    "lifetimes-waterice_99.0_+amorphouscarbon_1.0_-protopapa22dps.fits"
)

fig = plt.figure(1, (8, 4), clear=True)
axes = [fig.add_subplot(1, 4, i) for i in range(1, 5)]
for ax in axes:
    ax.minorticks_on()

ylim = [1e2, 1e12]

colors = [
    "skyblue",
    "mediumpurple",
    "violet",
    "tomato",
]

ymajor = []
yminor = []
yticklabels = []
labels = (
    "Pure",
    "0.2% carbon",
    "0.5% carbon",
    "1.0% carbon",
)
materials = (pure, ac002, ac005, ac010)
for i, (df, tau) in enumerate(zip(labels, materials)):
    for j in range(len(rh)):
        label = None
        axes[i].plot(a, tau[j], color=colors[i], ls="-", label=label, lw=1)

    axes[i].set_title(df, fontdict={"fontsize": 10})

    j = np.flatnonzero((tau[:, -1] > ylim[0]) * (tau[:, -1] < ylim[1]))
    ymajor.append(tau[j, -1])
    yticklabels.append(rh[j])

plt.setp(axes, xscale="log", yscale="log", xlim=[0.1, 50], ylim=ylim)
axes[1].set_xlabel("$a$ (μm)", x=1.1)
plt.setp(axes[1:4] + axes[5:], yticklabels=[])
axes[0].set_ylabel("$τ$ (s)")

for i in range(len(axes)):
    rax = axes[i].twinx()
    plt.setp(rax, ylim=ylim, yscale="log")
    rax.set_yticks(ymajor[i])
    rax.set_yticklabels(yticklabels[i])
    # rax.set_yticks(yminor[i], minor=True)
    # rax.set_yticklabels([], minor=True)

    if i == 3:
        rax.set_ylabel("$r_h$ (au)")

mskpy.niceplot(label_fs=12, tick_fs=9)
# fig.subplots_adjust(left=0.09, right=0.94, bottom=0.09, top=0.96, wspace=0.3)
plt.tight_layout()
plt.savefig("lifetimes-protopapa22dps.pdf")
os.system("pdf2ps lifetimes-protopapa22dps.pdf")


# add dynamical lifetimes
delta = np.array((5.4, 3.2, 2.5, 1.8, 1.9))
v = 20 * u.m / u.s
tau_d = (725 * u.km * delta / v).to_value("s")
for tau in tau_d:
    for ax in axes:
        ax.axhline(tau, color="k", ls="--", lw=1)
ax.minorticks_on()
plt.savefig("lifetimes-with-dynamical-protopapa22dps.pdf")
os.system("pdf2ps lifetimes-with-dynamical-protopapa22dps.pdf")
