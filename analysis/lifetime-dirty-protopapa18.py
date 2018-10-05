#import sys
# sys.path.append('/home/msk/local/python/')
import argparse

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from mskpy import davint
import grains2

parser = argparse.ArgumentParser()
parser.add_argument('dirt_fraction', type=float)
parser.add_argument('diameter', type=float, help='μm')
args = parser.parse_args()


def lifetime(a, m, rh):
    g = grains2.SublimationLTE(a, m, rh)

    dadt = g.dadt()
    tau0 = -a[0] / dadt[0]
    tau = [davint(a, -1.0 / dadt, a[0], b) for b in a[2:]]
    tau = tau0 + np.r_[0, -(a[1] - a[0]) / dadt[1], tau]

    # add sputtering from nominal solar wind
    Zsp = 1.1e8 * rh**-2  # molec/s/cm2, Mukai and Schwem 1981
    dadt -= 1e7 * Zsp * g.m.mu / g.m.rho
    tau_sp0 = -a[0] / dadt[0]
    tau_sp = [davint(a, -1.0 / dadt, a[0], b) for b in a[2:]]
    tau_sp = tau_sp0 + np.r_[0, -(a[1] - a[0]) / dadt[1], tau_sp]

    return tau, tau_sp


plt.clf()
a = np.logspace(np.log10(0.01 / 2), np.log10(args.diameter / 2))
emt = grains2.Bruggeman()
r = (1 - args.dirt_fraction, args.dirt_fraction)
mix = emt.mix((grains2.waterice(), grains2.amcarbon()), r,
              name='water ice({:.1%})+amorphous carbon({:.1%})'.format(*r))
dirtyice = grains2.waterice()
dirtyice.ri = mix.ri  # update n,k but use all other parameters
dirtyice.name = mix.name

print('dirt_fraction:', args.dirt_fraction)
print('radius:', args.diameter / 2)
print(grains2.waterice())
print(dirtyice)
for rh in [5.8, 5.0, 3.9, 2.3, 1.8, 1.3, 1.07]:
    print(rh)
    print('  pure')
    tau, tau_sp = lifetime(a, grains2.waterice(), rh)
    line = plt.plot(np.log10(a), np.log10(tau_sp), label=rh)[0]
    print('    {:.4g}'.format(tau_sp[-1]))

    print('  dirty ice')
    tau, tau_sp = lifetime(a, dirtyice, rh)
    plt.plot(np.log10(a), np.log10(tau_sp), color=line.get_color(), ls='--')
    print('    {:.4g}'.format(tau_sp[-1]))

plt.setp(plt.gca(), xlabel=r'log$_{10}$ $a$ (μm)', xlim=[-2, 0],
         ylabel=r'log$_{10}$ $\tau$ (s)')
plt.legend()
plt.show()
