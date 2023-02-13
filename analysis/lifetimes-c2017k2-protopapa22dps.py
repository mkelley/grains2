import numpy as np
from astropy.io import fits
from astropy.table import Table
import grains2

emt = grains2.Bruggeman()

ratios = [0, 0.002, 0.005, 0.01]

materials = [grains2.waterice()]
for r in ratios[1:]:
    if r > 0.01:
        raise ValueError("this is a small fraction approximation")

    mix = emt.mix(
        (grains2.waterice(), grains2.amcarbon()),
        (1 - r, r),
        name=f"water ice({1 - r:.1%})+amorphous carbon({r:.1%})",
    )
    dirtyice = grains2.waterice()
    dirtyice.ri = mix.ri  # update n,k but use all other parameters
    dirtyice.name = mix.name
    materials.append(dirtyice)

for r in ratios[1:]:
    mix = emt.mix(
        (grains2.waterice(), grains2.ampyroxene50()),
        (1 - r, r),
        name=f"water ice({1 - r:.1%})+amorphous pyroxene 50({r:.1%})",
    )
    dirtyice = grains2.waterice()
    dirtyice.ri = mix.ri  # update n,k but use all other parameters
    dirtyice.name = mix.name
    materials.append(dirtyice)

rh = np.array((5.4, 3.4, 3.2, 2.6, 2.5))
delta = np.array((5.4, 3.2, 2.5, 1.8, 1.9))
a = np.logspace(np.log10(0.01), np.log10(50))
tab = Table()
tab["a"] = a
tab["a"].unit = "um"
tab.meta["comments"] = "column labels are dirt fraction by volume"
for r, m in zip(ratios + ratios[1:], materials):
    tau = np.empty((len(rh), len(a)))
    for i in range(len(rh)):
        g = grains2.SublimationLTE(a, m, rh[i])
        tau[i] = g.lifetime()

    hdu = fits.HDUList()

    h = fits.Header()
    h["material"] = m.name
    h["bunit"] = "s"
    hdu.append(fits.PrimaryHDU(tau, h, "tau"))

    h = fits.Header()
    h["bunit"] = "au"
    hdu.append(fits.ImageHDU(rh, h, "rh"))

    h = fits.Header()
    h["bunit"] = "micrometer"
    hdu.append(fits.ImageHDU(a, h, "radius"))

    file_label = (
        m.name.replace("(", "_")
        .replace(")", "_")
        .replace("%", "")
        .replace(" ", "")
    )
    hdu.writeto(f"lifetimes-{file_label}-protopapa22dps.fits", overwrite=True)
