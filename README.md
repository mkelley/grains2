# grains2

grains2 enables modeling of thermal emission and ice sublimation in cometary comae.

WARNING: Thermal emission code is not well tested.  The temperature calculations are good, but the spectroscopic results have not been verified.

The sublimation model balances insolation with energy lost from sublimation and thermal radiation.  Losses due to solar wind sputtering (Mukai & Schwehm 1981) may also be included.  The optical constants of water ice from Warren & Brandt (2008) are included.  The latent heat of sublimation follows Delsemme & Miller (1971), and the vapor pressure equation of Lichtenegger & Komle (1991) is also used. Dust-ice aggregates may be created by mixing optical constants with the effective medium approximation of Bruggeman (1935).

Bohren and Huffman Mie and coated sphere code from Bruce Draine https://www.astro.princeton.edu/~draine/scattering.html  See source code for improvements on the original BH code.

davint is public domain

- Mike Kelley
