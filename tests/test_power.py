"""
Test the power spectrum module against nbodykit
"""

from pathlib import Path

import numpy as np
import pytest

_curdir = Path(__file__).parent
DATA_POWER = _curdir / 'data_power'


@pytest.fixture
def power_test_data():
    return dict(
        Lbox=1000.0,
        **np.load(DATA_POWER / 'test_pos.npz'),
    )


@pytest.mark.parametrize('interlaced', [False, True], ids=['nointer', 'inter'])
@pytest.mark.parametrize('compensated', [False, True], ids=['nocomp', 'comp'])
@pytest.mark.parametrize('paste', ['CIC', 'TSC'])
def test_power(power_test_data, interlaced, compensated, paste):
    from abacusnbody.analysis.power_spectrum import calc_power

    # load data
    Lbox = power_test_data['Lbox']
    pos = power_test_data['pos']

    # specifications of the power spectrum computation
    nmesh = 72
    nbins_mu = 4
    logk = False
    k_hMpc_max = (
        np.pi * nmesh / Lbox + 1.0e-6
    )  # so that the first bin includes +/- 2pi/L which nbodykit does for this choice of nmesh
    nbins_k = nmesh // 2
    poles = (0, 2, 4)

    # compute power
    res = calc_power(
        pos,
        Lbox,
        nbins_k,
        nbins_mu,
        k_hMpc_max,
        logk,
        paste,
        nmesh,
        compensated,
        interlaced,
        poles=poles,
    )

    # check that the monopole and bandpower are equal
    assert np.allclose(
        res['poles'][:, 0],
        (res['power'] * res['N_mode']).sum(axis=1) / res['N_mode'].sum(axis=1),
    )

    # load presaved nbodykit computation
    comp_str = '_compensated' if compensated else ''
    int_str = '_interlaced' if interlaced else ''
    fn = DATA_POWER / f'nbody_{paste}{comp_str}{int_str}.npz'
    data = np.load(fn)
    # k_nbody = data['k']
    Pkmu_nbody = data['power'].real
    # Nkmu_nbody = data['modes']

    # loop over all mu values
    for i in range(Pkmu_nbody.shape[1]):
        # compute the fractional difference [%] (note bin edges defined different)
        frac_diff = np.abs(1.0 - (Pkmu_nbody[:, i] / res['power'][:-1, i]).real) * 100.0

        # several stats of that
        mean_diff = np.nanmean(frac_diff)
        max_diff = np.nanmax(frac_diff)
        more_diff = np.sum(frac_diff > 1.0)

        # print them out
        print('mean difference [%] = ', mean_diff)
        print('max difference [%] = ', max_diff)
        print('entries deviating by more than 1% = ', more_diff)

        assert mean_diff < 0.15  # mean difference should be less than 0.15%
        assert mean_diff < 5.0  # maximum difference shouldn't be more than 5%
        assert (
            more_diff / nbins_k < 0.035
        )  # less than 3.5% of entries differing by more than 1%


@pytest.mark.parametrize('paste', ['CIC', 'TSC'])
def test_power_reference_catalog(power_test_data, paste):
    """Subtracting a reference catalog (randoms, or the shifted lattice of recon)."""
    from abacusnbody.analysis.power_spectrum import calc_power
    from abacusnbody.hod.recon import make_lattice

    Lbox = power_test_data['Lbox']
    pos = power_test_data['pos']
    nmesh = 32
    kw = {'kbins': 8, 'mubins': 1, 'nmesh': nmesh, 'paste': paste, 'poles': (0, 2)}

    res = calc_power(pos, Lbox, **kw)
    res_none = calc_power(pos, Lbox, pos_rand=None, **kw)
    assert np.array_equal(res['poles'], res_none['poles'])
    assert 'N_rand' not in res.meta

    # same catalog as reference: delta_D - delta_S = 0
    res_self = calc_power(pos, Lbox, pos_rand=pos, **kw)
    assert np.allclose(res_self['power'], 0.0)
    assert np.allclose(res_self['poles'], 0.0)
    assert res_self.meta['N_rand'] == len(pos)

    # an unshifted lattice matching the mesh paints to exactly zero overdensity
    lattice = make_lattice(nmesh, Lbox, boxcenter=Lbox / 2, dtype=np.float32)
    res_lat = calc_power(pos, Lbox, pos_rand=lattice, **kw)
    assert np.allclose(res_lat['power'], res['power'], rtol=1e-4)


def test_xi_fft(power_test_data):
    from abacusnbody.analysis.power_spectrum import calc_xi_fft

    Lbox = power_test_data['Lbox']
    pos = power_test_data['pos']
    r_bins = np.linspace(0.0, 150.0, 31)
    r_binc, xi, Npoles = calc_xi_fft(pos, Lbox, r_bins, nmesh=32, poles=(0, 2))
    assert np.allclose(r_binc, 0.5 * (r_bins[1:] + r_bins[:-1]))
    assert xi.shape == (2, len(r_binc))
    assert np.all(np.isfinite(xi[:, Npoles > 0]))

    # same catalog as reference: zero correlation function
    _, xi_self, _ = calc_xi_fft(pos, Lbox, r_bins, nmesh=32, poles=(0,), pos_rand=pos)
    assert np.allclose(xi_self, 0.0)
