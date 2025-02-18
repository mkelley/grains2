import numpy as np
from grains2 import material


class TestMaterial:
    def test_materials(self):
        m = material.amcarbon()
        m = material.amolivine40()
        m = material.amolivine50()
        m = material.ampyroxene40()
        m = material.ampyroxene50()
        m = material.magnetite()
        m = material.neutral(1.5 + 0.1 * 1j)
        m = material.olivine95()
        m = material.vacuum()
        m = material.waterice()

    def test_ri_copy(self):
        ri = material.amcarbon().ri
        copy = ri.copy()
        copy.nk *= 2
        assert not np.allclose(ri.nk, copy.nk)
        assert np.allclose(ri.nk * 2, copy.nk)
