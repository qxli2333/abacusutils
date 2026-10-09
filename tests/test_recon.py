"""
Tests of BAO reconstruction of HOD catalogs (`AbacusHOD.run_recon`) and of the
post-reconstruction clustering measurements with linear control variates (LCV).

The reconstruction tests require pyrecon:
    $ pip install git+https://github.com/cosmodesi/pyrecon@main
"""

import shutil
from os.path import dirname
from os.path import join as pjoin

import asdf
import numba
import numpy as np
import pytest
import yaml

# required for pytest to work (see GH #60)
numba.config.THREADING_LAYER = 'forksafe'

TESTDIR = dirname(__file__)
EXAMPLE_CONFIG = pjoin(TESTDIR, 'abacus_hod.yaml')
ZCV_DATA = pjoin(TESTDIR, 'data_zcv', 'AbacusSummit_base_c000_ph006')

ALGORITHMS = [
    'IterativeFFTReconstruction',
    'MultiGridReconstruction',
    'IterativeFFTParticleReconstruction',
]
RECON_PARAMS = {'nmesh': 16, 'smoothing_radius': 4.0}


def test_recon_params():
    from abacusnbody.hod.recon import (
        DEFAULT_RECON_BIAS,
        DEFAULT_RECON_PARAMS,
        DEFAULT_RECON_SMOOTHING_RADIUS,
        _get_per_tracer,
        get_recon_params,
    )

    params = get_recon_params(None)
    assert params == DEFAULT_RECON_PARAMS
    assert params['bias'] == DEFAULT_RECON_BIAS
    assert params['bias'] is not DEFAULT_RECON_PARAMS['bias']
    assert params['smoothing_radius'] == {'LRG': 15.0, 'ELG': 15.0, 'QSO': 30.0}
    assert params['smoothing_radius'] is not DEFAULT_RECON_SMOOTHING_RADIUS
    assert (
        _get_per_tracer(params['smoothing_radius'], 'QSO', 'smoothing_radius') == 30.0
    )
    assert _get_per_tracer(10.0, 'QSO', 'smoothing_radius') == 10.0
    with pytest.raises(KeyError):
        _get_per_tracer({'LRG': 15.0}, 'ELG', 'smoothing_radius')

    params = get_recon_params({'convention': 'RecIso', 'bias': 1.5})
    assert params['convention'] == 'reciso'
    assert params['bias'] == 1.5

    # cellsize replaces the default nmesh
    params = get_recon_params({'cellsize': 4.0})
    assert params['nmesh'] is None and params['cellsize'] == 4.0

    with pytest.raises(ValueError):
        get_recon_params({'nmesh': 64, 'cellsize': 4.0})
    with pytest.raises(ValueError):
        get_recon_params({'convention': 'recfoo'})
    with pytest.raises(ValueError):
        get_recon_params({'smoothing': 15.0})  # typo of smoothing_radius
    for radius in (0.0, {'LRG': 15.0, 'QSO': -1.0}):
        with pytest.raises(ValueError):
            get_recon_params({'smoothing_radius': radius})
    with pytest.raises(NotImplementedError):
        get_recon_params({'engine': 'foo'})


def test_make_lattice():
    from abacusnbody.hod.recon import make_lattice

    n, Lbox = 6, 12.0
    lattice = make_lattice(n, Lbox)
    assert lattice.shape == (n**3, 3)
    assert np.allclose(np.unique(lattice[:, 0]), (np.arange(n) + 0.5) * 2.0 - 6.0)
    # slab by slab gives the same lattice
    slabs = [make_lattice(n, Lbox, islab=slice(i, i + 4)) for i in range(0, n, 4)]
    assert np.array_equal(np.concatenate(slabs), lattice)


@pytest.fixture(scope='module')
def hod(tmp_path_factory):
    """AbacusHOD object and RSD mock on the Mini_N64_L32 test simulation."""
    pytest.importorskip('pyrecon')
    from abacusnbody.hod import prepare_sim
    from abacusnbody.hod.abacus_hod import AbacusHOD

    tmp_path = tmp_path_factory.mktemp('recon')
    with open(EXAMPLE_CONFIG) as fp:
        config = yaml.safe_load(fp)
    config['sim_params']['sim_dir'] = TESTDIR
    config['sim_params']['output_dir'] = pjoin(tmp_path, 'data_mocks') + '/'
    config['sim_params']['subsample_dir'] = pjoin(tmp_path, 'data_subs') + '/'
    prepare_sim.main(EXAMPLE_CONFIG, params=config)

    ball = AbacusHOD(
        config['sim_params'], config['HOD_params'], config['clustering_params']
    )
    mock_dict = ball.run_hod(ball.tracers, want_rsd=True, Nthread=2)
    return {
        'ball': ball,
        'mock_dict': mock_dict,
        'config': config,
        'tmp_path': tmp_path,
    }


@pytest.mark.parametrize('convention', ['recsym', 'reciso'])
@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_run_recon(hod, algorithm, convention):
    pytest.importorskip('pyrecon')
    ball, mock_dict = hod['ball'], hod['mock_dict']
    Lbox = ball.lbox
    params = dict(RECON_PARAMS, algorithm=algorithm, convention=convention)

    recon_dict = ball.run_recon(mock_dict, params, Nthread=2)
    assert recon_dict.keys() == mock_dict.keys()
    for tr in mock_dict:
        rec = recon_dict[tr]
        info = rec['recon_info']
        assert info['convention'] == convention
        assert info['f'] == pytest.approx(ball.params['f_growth'])
        assert info['bias'] == params.get('bias', {'LRG': 2.0, 'ELG': 1.2}[tr])
        # other galaxy properties are carried over
        assert np.array_equal(rec['mass'], mock_dict[tr]['mass'])

        pos = np.stack([rec[ax] for ax in 'xyz'], axis=1)
        pos_S = np.stack([rec['shifted'][ax] for ax in 'xyz'], axis=1)
        assert pos.shape == (len(mock_dict[tr]['x']), 3)
        assert pos_S.shape == (16**3, 3)  # lattice_nmesh defaults to nmesh
        for p in (pos, pos_S):
            assert p.dtype == np.float32
            assert np.all((p >= -Lbox / 2) & (p < Lbox / 2))
        pos_in = np.stack([mock_dict[tr][ax] for ax in 'xyz'], axis=1)
        assert not np.allclose(pos, pos_in)

    # lattice is deterministic
    recon_dict2 = ball.run_recon(mock_dict, params, Nthread=2)
    for tr in mock_dict:
        assert np.array_equal(recon_dict[tr]['x'], recon_dict2[tr]['x'])
        assert np.array_equal(
            recon_dict[tr]['shifted']['z'], recon_dict2[tr]['shifted']['z']
        )


def test_recon_shifted_field(hod):
    pytest.importorskip('pyrecon')
    from abacusnbody.hod.recon import run_recon_pyrecon

    ball, mock_dict = hod['ball'], hod['mock_dict']
    tr = 'ELG'
    pos = np.stack([mock_dict[tr][ax] for ax in 'xyz'], axis=1)
    kw = {'smoothing_radius': 4.0, 'nmesh': 16, 'nthread': 2}

    # without RSD removal (f = 0), RecSym and RecIso shift the lattice identically
    _, S_sym = run_recon_pyrecon(pos, ball.lbox, 0.0, 1.2, convention='recsym', **kw)
    _, S_iso = run_recon_pyrecon(pos, ball.lbox, 0.0, 1.2, convention='reciso', **kw)
    assert np.array_equal(S_sym, S_iso)
    # with RSD removal they differ only along the line of sight
    _, S_sym = run_recon_pyrecon(pos, ball.lbox, 0.8, 1.2, convention='recsym', **kw)
    _, S_iso = run_recon_pyrecon(pos, ball.lbox, 0.8, 1.2, convention='reciso', **kw)
    assert np.array_equal(S_sym[:, :2], S_iso[:, :2])
    assert not np.allclose(S_sym[:, 2], S_iso[:, 2])

    # shifting the lattice slab by slab gives the same result
    _, S_slabs = run_recon_pyrecon(
        pos, ball.lbox, 0.8, 1.2, convention='recsym', max_slab_points=16**2 * 3, **kw
    )
    assert np.array_equal(S_slabs, S_sym)

    # randoms instead of a lattice
    kw_rand = dict(kw, shifted_field='randoms', nrandoms_factor=5, seed=7)
    _, R1 = run_recon_pyrecon(pos, ball.lbox, 0.8, 1.2, **kw_rand)
    _, R2 = run_recon_pyrecon(pos, ball.lbox, 0.8, 1.2, **kw_rand)
    assert R1.shape == (5 * len(pos), 3)
    assert np.array_equal(R1, R2)


def test_recon_real_space_and_bias(hod):
    pytest.importorskip('pyrecon')
    ball, mock_dict = hod['ball'], hod['mock_dict']

    recon_dict = ball.run_recon(
        mock_dict, dict(RECON_PARAMS, bias=1.7), want_rsd=False, Nthread=2
    )
    for tr in mock_dict:
        assert recon_dict[tr]['recon_info']['f'] == 0.0
        assert recon_dict[tr]['recon_info']['bias'] == 1.7

    with pytest.raises(KeyError):
        ball.run_recon(mock_dict, dict(RECON_PARAMS, bias={'LRG': 2.0}))


def test_recon_per_tracer_smoothing(hod):
    pytest.importorskip('pyrecon')
    ball, mock_dict = hod['ball'], hod['mock_dict']

    radius = {'LRG': 4.0, 'ELG': 6.0}
    recon_dict = ball.run_recon(
        mock_dict, dict(RECON_PARAMS, smoothing_radius=radius), Nthread=2
    )
    for tr in mock_dict:
        assert recon_dict[tr]['recon_info']['smoothing_radius'] == radius[tr]
    # the LRG reconstruction used the LRG radius
    lrg = {'LRG': mock_dict['LRG']}
    for R, same in ((4.0, True), (6.0, False)):
        rec = ball.run_recon(lrg, dict(RECON_PARAMS, smoothing_radius=R), Nthread=2)
        assert np.array_equal(rec['LRG']['x'], recon_dict['LRG']['x']) == same

    with pytest.raises(KeyError):
        ball.run_recon(mock_dict, dict(RECON_PARAMS, smoothing_radius={'LRG': 4.0}))


def test_rec_settings_per_tracer():
    tools_cv = pytest.importorskip('abacusnbody.hod.zcv.tools_cv')

    config = {
        'HOD_params': {'tracer_flags': {'LRG': False, 'ELG': True, 'QSO': False}},
        'recon_params': {
            'convention': 'reciso',
            'smoothing_radius': {'LRG': 15.0, 'ELG': 10.0, 'QSO': 30.0},
        },
    }
    assert tools_cv._get_rec_settings(config) == ('reciso', 10.0)
    config['recon_params']['convention'] = 'recsym'
    assert tools_cv._get_rec_settings(config) == ('recsym', None)


def test_recon_clustering(hod):
    pytest.importorskip('pyrecon')
    ball, mock_dict, config = hod['ball'], hod['mock_dict'], hod['config']

    recon_dict = ball.run_recon(mock_dict, RECON_PARAMS, Nthread=2)

    # power spectrum of delta_D - delta_S
    clustering = ball.compute_power(
        recon_dict, 4, 1, 0.8, False, poles=[0, 2], num_cells=16
    )
    assert np.all(np.isfinite(clustering['LRG_ELG_ell']))

    # pair-count estimators ignore delta_S
    with pytest.raises(ValueError):
        ball.compute_xirppi(recon_dict, ball.rpbins, 30, 5, Nthread=2)
    with pytest.raises(ValueError):
        ball.compute_wp(recon_dict, ball.rpbins, 30, 5, Nthread=2)

    # raw measurements (no control variates), single tracer
    single = {'ELG': recon_dict['ELG']}
    config = dict(config, power_params=dict(config['power_params'], nmesh=16))
    pk = ball.apply_cv(single, config, stat='pk', cv_type=None)
    assert pk['Pk_tr_tr_ell'].shape == (3, 4)
    xi = ball.apply_cv(single, config, stat='xi', cv_type=None)
    assert xi['Xi_tr_tr_ell'].shape == (3, len(xi['r_binc']))

    # ZCV is not defined for reconstructed catalogs, and LCV only for them
    with pytest.raises(NotImplementedError):
        ball.apply_cv(single, config, cv_type='zcv')
    with pytest.raises(NotImplementedError):
        ball.apply_cv({'ELG': mock_dict['ELG']}, config, cv_type='lcv')


def test_recon_lcv(hod, tmp_path):
    """Smoke test of the default (LCV) control variates after reconstruction."""
    pytest.importorskip('pyrecon')
    pytest.importorskip('classy')
    from abacusnbody.hod.zcv import linear_fields

    ball, mock_dict = hod['ball'], hod['mock_dict']
    config = yaml.safe_load(yaml.safe_dump(hod['config']))

    # inputs of the linear fields (filtered ICs and window function)
    sim_name = 'AbacusSummit_base_c000_ph006'  # so that meta can find it
    lcv_dir = tmp_path / 'lcv'
    (lcv_dir / sim_name).mkdir(parents=True)
    for fn in ('ic_filt_nmesh8.asdf', 'window_nmesh8.npz'):
        shutil.copy(pjoin(ZCV_DATA, fn), lcv_dir / sim_name / fn)
    config['sim_params']['sim_name'] = sim_name
    config['sim_params']['z_mock'] = 0.8  # so that meta can find it
    config['lcv_params']['lcv_dir'] = str(lcv_dir)
    config_fn = tmp_path / 'config.yaml'
    with open(config_fn, 'w') as fp:
        yaml.safe_dump(config, fp)
    linear_fields.main(str(config_fn))
    linear_fields.main(str(config_fn), save_3D_power=True)
    # the 3D run must not overwrite the binned power spectra
    with asdf.open(lcv_dir / sim_name / 'power_lin_nmesh8.asdf') as af:
        assert 'P_ell_delta_delta' in af['data']

    recon_dict = ball.run_recon(
        {'LRG': mock_dict['LRG']}, config['recon_params'], Nthread=2
    )
    lcv_dict = ball.apply_cv(recon_dict, config, stat='pk')
    for key in ('k_binc', 'Pk_tr_tr_ell', 'Pk_tr_tr_ell_lcv', 'rho_tr_lf', 'bias'):
        assert key in lcv_dict
    assert lcv_dict['Pk_tr_tr_ell_lcv'].shape == (3, 4)

    # reuse the saved tracer power spectra
    lcv_dict2 = ball.apply_cv(recon_dict, config, stat='pk', load_presaved=True)
    assert np.allclose(lcv_dict2['Pk_tr_tr_ell'], lcv_dict['Pk_tr_tr_ell'])

    # RecIso
    config['recon_params']['convention'] = 'reciso'
    recon_iso = ball.run_recon(
        {'LRG': mock_dict['LRG']}, config['recon_params'], Nthread=2
    )
    lcv_iso = ball.apply_cv(recon_iso, config, stat='pk')
    assert lcv_iso['Pk_tr_tr_ell_lcv'].shape == (3, 4)

    lcv_xi = ball.apply_cv(recon_dict, config, stat='xi')
    for key in ('r_binc', 'Xi_tr_tr_ell', 'Xi_tr_tr_ell_lcv'):
        assert key in lcv_xi
    assert lcv_xi['Xi_tr_tr_ell_lcv'].shape == (3, len(lcv_xi['r_binc']))
