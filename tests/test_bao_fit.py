"""
Tests of the BAO fitting module `abacusnbody.hod.bao_fit`: Gaussian covariances of the
power spectrum and correlation function multipoles of a periodic box (checked against
Monte Carlo realizations of Gaussian random fields), and recovery of the BAO dilation
parameters with desilike.

The desilike tests require desilike, cosmoprimo and lsstypes:
    $ pip install git+https://github.com/cosmodesi/desilike
"""

import os

import numba
import numpy as np
import pytest

# required for pytest to work (see GH #60)
numba.config.THREADING_LAYER = 'forksafe'
# desilike imports mpi4py; allow OpenMPI singletons when running as root (containers)
os.environ.setdefault('OMPI_ALLOW_RUN_AS_ROOT', '1')
os.environ.setdefault('OMPI_ALLOW_RUN_AS_ROOT_CONFIRM', '1')


def test_get_bao_params():
    from abacusnbody.hod.bao_fit import DEFAULT_BAO_PARAMS, get_bao_params

    params = get_bao_params(None)
    assert params['stat'] == 'xi' and params['broadband'] == 'pcs2'
    assert params['slim'] == (60.0, 150.0, 4.0)
    assert DEFAULT_BAO_PARAMS['broadband'] is None
    assert get_bao_params({'stat': 'pk'})['broadband'] == 'pcs'
    assert get_bao_params({'ells': [0, 2]})['ells'] == (0, 2)
    for bad in ({'stat': 'bk'}, {'apmode': 'qfoo'}, {'method': 'mcmc'}, {'foo': 1}):
        with pytest.raises(ValueError):
            get_bao_params(bad)


def test_rebin_xi_poles():
    from abacusnbody.hod.bao_fit import rebin_xi_poles

    r = np.arange(0.5, 20.0)
    counts = r**2
    xi = np.array([np.ones_like(r), r])
    s, xi_rebinned = rebin_xi_poles(r, xi, counts, np.arange(4.0, 17.0, 4.0))
    assert xi_rebinned.shape == (2, 3)
    assert np.allclose(xi_rebinned[0], 1.0)
    assert np.allclose(xi_rebinned[1], s)  # linear function: average = weighted center
    with pytest.raises(ValueError):
        rebin_xi_poles(r, xi, np.zeros_like(r), np.arange(4.0, 17.0, 4.0))


def test_cv_variance_ratio():
    pytest.importorskip('classy')
    from abacusnbody.hod.zcv.tools_cv import cv_variance_ratio

    # perfectly correlated control variate with beta = 1: no variance left
    ratio = cv_variance_ratio(np.ones(3), np.ones(3), np.ones(3), np.ones(3))
    assert np.allclose(ratio, 0.0)
    # beta = 0: no reduction; undefined entries default to 1
    ratio = cv_variance_ratio(np.zeros(3), np.array([1.0, 1.0, 0.0]), 0.5, 1.0)
    assert np.allclose(ratio, 1.0)


def _gaussian_field_realizations(L, N, nreal, shot, seed=0):
    """Gaussian random fields with a Kaiser-like anisotropic P(k, mu) plus shot noise."""
    from scipy.fft import rfftn

    rng = np.random.default_rng(seed)
    kf = 2 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        MU = np.where(K > 0, KZ / K, 0.0)
    plin = 2e4 * (K / 0.02) / (1 + (K / 0.02) ** 2.5)
    pkmu = (2.0 + 0.7 * MU**2) ** 2 * plin + shot
    pkmu[0, 0, 0] = 0.0
    amplitude = np.sqrt(pkmu * N**3 / L**3) / N**3
    for _ in range(nreal):
        wk = rfftn(rng.standard_normal((N, N, N)))
        yield (wk * amplitude).astype(np.complex64)


def test_gaussian_covariance_pk_poles():
    from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak
    from abacusnbody.hod.bao_fit import gaussian_covariance_pk_poles

    L, N, poles = 600.0, 32, (0, 2, 4)
    kf = 2 * np.pi / L
    kedges = np.arange(0.5, N // 2 + 1) * kf
    pks = []
    for field_fft in _gaussian_field_realizations(L, N, 400, 3000.0):
        P = calc_pk_from_deltak(
            field_fft, L, kedges, np.array([0.0, 1.0]), poles=np.asarray(poles)
        )
        pks.append(P['binned_poles'])
    pks = np.array(pks)
    mask = kedges[:-1] > 0.03
    flat = np.array([np.concatenate([p[0][mask], p[1][mask]]) for p in pks])
    cov = gaussian_covariance_pk_poles(
        pks.mean(axis=0), P['N_mode_poles'], poles, ells=(0, 2)
    )
    cov = cov[np.ix_(np.tile(mask, 2), np.tile(mask, 2))]
    diff = flat - flat.mean(axis=0)
    chi2 = np.einsum('ij,jk,ik->i', diff, np.linalg.inv(cov), diff).mean()
    assert chi2 / flat.shape[1] == pytest.approx(1.0, abs=0.1)

    # control variates: the covariance scales with the variance ratio
    ratio = np.full((len(poles), len(kedges) - 1), 0.25)
    cov_cv = gaussian_covariance_pk_poles(
        pks.mean(axis=0), P['N_mode_poles'], poles, ells=(0, 2), variance_ratio=ratio
    )
    cov_cv = cov_cv[np.ix_(np.tile(mask, 2), np.tile(mask, 2))]
    assert np.allclose(cov_cv, 0.25 * cov)


def test_gaussian_covariance_xi_poles():
    from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak, pk_to_xi
    from abacusnbody.hod.bao_fit import gaussian_covariance_xi_poles, rebin_xi_poles

    # mesh cells (3.9 Mpc/h) smaller than the separation bins (4 Mpc/h)
    L, N, poles = 250.0, 64, (0, 2, 4)
    kf = 2 * np.pi / L
    kedges = np.arange(0.5, N // 2 + 1) * kf
    s_edges = np.arange(40.0, 101.0, 4.0)
    pks, xis = [], []
    for field_fft in _gaussian_field_realizations(L, N, 300, 2000.0, seed=1):
        P = calc_pk_from_deltak(
            field_fft, L, kedges, np.array([0.0, 1.0]), poles=np.asarray(poles)
        )
        pks.append(P['binned_poles'])
        pk3d = np.asarray((field_fft * np.conj(field_fft)).real, dtype=np.float32)
        r_binc, xi, counts = pk_to_xi(pk3d, L, np.linspace(0, 120, 121), poles=poles)
        xis.append(rebin_xi_poles(r_binc, xi, counts, s_edges)[1][:2].ravel())
    xis = np.array(xis)
    k = 0.5 * (kedges[1:] + kedges[:-1])
    cov = gaussian_covariance_xi_poles(
        s_edges, k, np.mean(pks, axis=0), poles, L, N, ells=(0, 2)
    )
    diff = xis - xis.mean(axis=0)
    chi2 = np.einsum('ij,jk,ik->i', diff, np.linalg.inv(cov), diff).mean()
    assert chi2 / xis.shape[1] == pytest.approx(1.0, abs=0.2)
    ratio = np.diag(np.cov(xis, rowvar=False)) / np.diag(cov)
    assert np.mean(ratio) == pytest.approx(1.0, abs=0.25)


@pytest.fixture(scope='module')
def bao_mock():
    """Noiseless BAO multipoles from the desilike model itself, with known dilations."""
    pytest.importorskip('desilike')
    from cosmoprimo.fiducial import AbacusSummit
    from desilike.theories.galaxy_clustering import (
        BAOPowerSpectrumTemplate,
        DampedBAOWigglesTracerCorrelationFunctionMultipoles,
        DampedBAOWigglesTracerPowerSpectrumMultipoles,
    )

    L, z, nmesh, nbar = 2000.0, 0.8, 256, 5e-4
    kf = 2 * np.pi / L
    k = (np.arange(nmesh // 2) + 0.5) * kf
    truth = {'qiso': 1.012, 'qap': 0.985, 'b1': 2.0, 'sigmapar': 6.0, 'sigmaper': 3.0}
    cosmo = AbacusSummit(name='000', engine='eisenstein_hu')
    template = BAOPowerSpectrumTemplate(z=z, fiducial=cosmo, apmode='qisoqap')
    kwargs = {'template': template, 'mode': 'recsym', 'ells': (0, 2, 4)}
    pk = np.array(
        DampedBAOWigglesTracerPowerSpectrumMultipoles(k=k, broadband='pcs', **kwargs)(
            **truth
        )
    )
    r_binc = np.arange(0.5, 200)
    xi = np.array(
        DampedBAOWigglesTracerCorrelationFunctionMultipoles(
            s=r_binc, broadband='pcs2', **kwargs
        )(**truth)
    )
    cv_dict = {
        'poles': (0, 2, 4),
        'k_binc': k,
        'Pk_tr_tr_ell': pk + np.array([1 / nbar, 0, 0])[:, None],
        'Nk_tr_tr_ell': 4 * np.pi * k**2 / kf**2,
        'r_binc': r_binc,
        'Xi_tr_tr_ell': xi,
        'Np_tr_tr_ell': 4 * np.pi * r_binc**2,
    }
    return {
        'cv_dict': cv_dict,
        'truth': truth,
        'z': z,
        'Lbox': L,
        'nbar': nbar,
        'nmesh': nmesh,
    }


@pytest.mark.parametrize('stat', ['pk', 'xi'])
def test_fit_bao_recovery(bao_mock, stat):
    from abacusnbody.hod.bao_fit import fit_bao

    result = fit_bao(
        bao_mock['cv_dict'],
        z=bao_mock['z'],
        Lbox=bao_mock['Lbox'],
        nbar=bao_mock['nbar'],
        tracer='LRG',
        recon_info={'convention': 'recsym', 'smoothing_radius': 15.0},
        sim_name='AbacusSummit_base_c000_ph006',
        nmesh=bao_mock['nmesh'],
        bao_params={
            'stat': stat,
            'data': 'raw',
            'engine': 'eisenstein_hu',
        },
    )
    truth = bao_mock['truth']
    assert result['params'] == ['qiso', 'qap']
    assert result['qiso'] == pytest.approx(truth['qiso'], abs=1e-3)
    assert result['qap'] == pytest.approx(truth['qap'], abs=3e-3)
    assert 0 < result['qiso_err'] < 0.02
    assert result['chi2'] < 0.1  # noiseless data
    assert result['data'].shape == result['model'].shape
    assert result['ndof'] > 0


def test_abacushod_fit_bao(bao_mock):
    """AbacusHOD.fit_bao wiring: redshift, box size, shot noise and recon settings."""
    from abacusnbody.hod.abacus_hod import AbacusHOD

    # only the attributes used by fit_bao (no simulation to load)
    ball = AbacusHOD.__new__(AbacusHOD)
    ball.sim_name = 'AbacusSummit_base_c000_ph006'
    ball.z_mock = bao_mock['z']
    ball.lbox = bao_mock['Lbox']
    ngal = int(bao_mock['nbar'] * ball.lbox**3)
    mock_dict = {
        'LRG': {
            'x': np.zeros(ngal, dtype=np.float32),
            'recon_info': {'convention': 'recsym', 'smoothing_radius': 15.0},
        }
    }
    config = {
        'power_params': {'nmesh': bao_mock['nmesh']},
        'bao_params': {
            'stat': 'pk',
            'data': 'raw',
            'engine': 'eisenstein_hu',
        },
    }
    result = ball.fit_bao(mock_dict, config, cv_dict=bao_mock['cv_dict'])
    assert result['mode'] == 'recsym' and result['z'] == bao_mock['z']
    assert result['qiso'] == pytest.approx(bao_mock['truth']['qiso'], abs=1e-3)


def test_fit_bao_control_variates(bao_mock):
    """With control variates, the reduced data vector and covariance are used."""
    from abacusnbody.hod.bao_fit import fit_bao

    raw = bao_mock['cv_dict']
    cv_dict = dict(raw)
    cv_dict['Pk_tr_tr_ell_lcv'] = raw['Pk_tr_tr_ell']
    cv_dict['Pk_tr_tr_ell'] = raw['Pk_tr_tr_ell'] * 1.1  # raw differs from LCV
    cv_dict['cv_variance_ratio'] = np.full(np.shape(raw['Pk_tr_tr_ell']), 0.25)
    kwargs = {
        'z': bao_mock['z'],
        'Lbox': bao_mock['Lbox'],
        'nbar': bao_mock['nbar'],
        'recon_info': {'convention': 'recsym', 'smoothing_radius': 15.0},
        'sim_name': 'AbacusSummit_base_c000_ph006',
    }
    params = {'stat': 'pk', 'engine': 'eisenstein_hu', 'niterations': 1}
    result = fit_bao(cv_dict, bao_params=params, **kwargs)
    result_nored = fit_bao(
        cv_dict, bao_params={**params, 'cov_cv_reduction': False}, **kwargs
    )
    assert np.allclose(result['covariance'], 0.25 * result_nored['covariance'])
    assert result['qiso'] == pytest.approx(bao_mock['truth']['qiso'], abs=2e-3)
    assert result['qiso_err'] == pytest.approx(0.5 * result_nored['qiso_err'], rel=0.1)


def test_fit_bao_emcee(bao_mock):
    """Sampling with the broadband marginalized analytically."""
    from abacusnbody.hod.bao_fit import fit_bao

    result = fit_bao(
        bao_mock['cv_dict'],
        z=bao_mock['z'],
        Lbox=bao_mock['Lbox'],
        nbar=bao_mock['nbar'],
        recon_info={'convention': 'recsym', 'smoothing_radius': 15.0},
        sim_name='AbacusSummit_base_c000_ph006',
        bao_params={
            'stat': 'pk',
            'engine': 'eisenstein_hu',
            'method': 'emcee',
            'run_kwargs': {'max_steps': 600},
        },
    )
    assert 'chain' in result
    assert result['qiso'] == pytest.approx(bao_mock['truth']['qiso'], abs=0.01)
    assert 0 < result['qiso_err'] < 0.02
