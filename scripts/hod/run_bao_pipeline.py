"""
Full BAO pipeline for one AbacusSummit phase and several HOD random seeds:
populate galaxies -> (pre-recon clustering + BAO fit) -> reconstruction ->
(post-recon clustering + BAO fit) -> save the BAO parameters and their errors.

The HOD parameters, the phase and the redshift are the same for all seeds; only the
random numbers used to populate the galaxies change (``run_hod(reseed=seed)``).
The BAO fits use desilike with Minuit profiling (``bao_params['method'] = 'profile'``).

Prerequisites (run once per phase and redshift, e.g. in a separate job):

* ``prepare_sim`` subsamples for the phase and redshift,
* the control-variate inputs, see ``abacusnbody.hod.zcv.ic_fields`` and
  ``abacusnbody.hod.zcv.linear_fields`` (ZCV before and LCV after reconstruction),
* pyrecon and desilike. The abacusutils of the shared DESI environment predates the
  reconstruction and BAO code: run with this repository first on ``PYTHONPATH``.

Outputs, in ``outdir``, for each seed:

* ``bao_<simname>_z<z>_seed<seed>.json``: qiso, qap, errors, chi2, ndof, all fitted
  parameters and errors, and the timing and memory of each stage,
* ``bao_<simname>_z<z>_seed<seed>.npz``: the measured multipoles (raw and with control
  variates), and the fitted data, covariance and best-fit model, pre- and post-recon.

Existing outputs are skipped (``--overwrite`` to redo them).

Usage
-----
$ python ./run_bao_pipeline.py --phase 0 --z 0.725 --seeds 1 2 3
"""

import argparse
import json
import os
import re
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod_lrg2_dr2v2.yaml'
DEFAULTS['outdir'] = '$SCRATCH/catalog/abacus/bao_fits'

SCALARS = ('qiso', 'qiso_err', 'qap', 'qap_err', 'chi2', 'ndof')


def _peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


class Timer:
    """Record the wall time and the peak memory so far of each stage."""

    def __init__(self):
        self.info = {}

    def __call__(self, name):
        timer = self

        class _Stage:
            def __enter__(self):
                self.start = time.time()

            def __exit__(self, *args):
                timer.info[name] = {
                    'time_s': time.time() - self.start,
                    'peak_rss_gb': _peak_rss_gb(),
                }
                print(
                    f'[{name}] {timer.info[name]["time_s"]:.1f} s, '
                    f'peak {timer.info[name]["peak_rss_gb"]:.1f} GB',
                    flush=True,
                )

        return _Stage()


# files written by the control-variate code (tracer fields and powers); the prepared inputs have none of these names
_TRACER_FILE = re.compile(r'(_tr_|_tr\.|tr_field|_ZCV_|_LCV_)')


def make_cv_workdir(config, workdir):
    """
    Give this process a private copy (symlinks) of the prepared control-variate inputs.

    The control-variate code writes the tracer fields and powers of each measurement to fixed
    file names next to the prepared inputs, so processes sharing a phase directory (e.g. one
    process per seed) would overwrite each other. Returns the list of directories created.
    """
    simname = config['sim_params']['sim_name']
    made = {}
    for block, key in (('zcv_params', 'zcv_dir'), ('lcv_params', 'lcv_dir')):
        if block not in config:
            continue
        src_root = Path(os.path.expandvars(config[block][key]))
        if str(src_root) not in made:
            dst_root = Path(workdir) / f'{block}_{len(made)}'
            src, dst = src_root / simname, dst_root / simname
            for fn in src.rglob('*'):
                rel = fn.relative_to(src)
                if fn.is_dir():
                    (dst / rel).mkdir(parents=True, exist_ok=True)
                    continue
                if _TRACER_FILE.search(fn.name):
                    raise RuntimeError(f'{fn} would be overwritten by tracer outputs')
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                (dst / rel).symlink_to(fn)
            made[str(src_root)] = dst_root
        config[block][key] = str(made[str(src_root)])
    return list(made.values())


def make_nfw_draw(seed, n=10_000_000, xmax=50.0):
    """
    Draw ``n`` values of x = r / r_s from the NFW profile, p(x) ~ x / (1 + x)^2 for x < xmax.

    ``run_hod(want_nfw=True)`` needs this array: each satellite takes a draw and rejects it
    if x > c (the halo concentration), which leaves the NFW distribution truncated at the
    virial radius. xmax only needs to exceed the largest concentration.
    """
    cdf = lambda x: np.log1p(x) + 1.0 / (1.0 + x) - 1.0  # integral of x / (1 + x)^2 from 0
    grid = np.concatenate([[0.0], np.geomspace(1e-6, xmax, 20000)])
    u = np.random.default_rng(seed).uniform(0.0, cdf(xmax), n)
    return np.interp(u, cdf(grid), grid)


def measure_and_fit(
    newBall, mock_dict, config, stats, cv_type, prefix, timer, out, hod_kwargs=None
):
    """Measure and fit each statistic; add scalars to ``out['json']`` and arrays to ``out['npz']``."""
    for stat in stats:
        bao_params = {**(config.get('bao_params') or {}), 'stat': stat}
        bao_params['method'] = 'profile'  # Minuit profiling
        tag = f'{prefix}_{stat}'
        with timer(f'{tag}_measure'):
            cv_dict = newBall.apply_cv(
                mock_dict, config, stat=stat, cv_type=cv_type, hod_kwargs=hod_kwargs
            )
        for key, val in cv_dict.items():
            val = np.asarray(val)
            if val.dtype != object and val.ndim > 0:
                out['npz'][f'{tag}_meas_{key}'] = val
        with timer(f'{tag}_fit'):
            result = newBall.fit_bao(
                mock_dict, config, cv_dict=cv_dict, bao_params=bao_params
            )
        out['json'][tag] = {key: float(result[key]) for key in SCALARS if key in result}
        out['json'][tag]['bestfit'] = result['bestfit']
        out['json'][tag]['error'] = result['error']
        for key in ('data', 'cov', 'model'):
            if key in result:
                out['npz'][f'{tag}_fit_{key}'] = np.asarray(result[key])
        print(
            f'{tag}: qiso = {result["qiso"]:.4f} +- {result["qiso_err"]:.4f}, '
            f'qap = {result.get("qap", np.nan):.4f} +- {result.get("qap_err", np.nan):.4f}, '
            f'chi2/ndof = {result["chi2"]:.1f}/{result["ndof"]}',
            flush=True,
        )


def run_seed(newBall, config, tracer, seed, stats, want_recon, cv_pre, cv_post, outfn):
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    recon_params = config.get('recon_params') or {}
    timer = Timer()
    out = {
        'json': {
            'simname': sim_params['sim_name'],
            'z': sim_params['z_mock'],
            'seed': seed,
            'tracer': tracer,
        },
        'npz': {},
    }

    # secondary redshifts have no particle subsample: satellites must follow NFW
    want_nfw = HOD_params.get('want_nfw', False) or newBall.z_type == 'secondary'
    # ZCV regenerates the same catalog without RSD: it must have the same seed and NFW settings
    hod_kwargs = {
        'want_nfw': want_nfw,
        'NFW_draw': make_nfw_draw(seed) if want_nfw else None,
        'reseed': seed,
        'Nthread': config.get('Nthread', 16),
    }
    with timer('hod'):
        mock_dict = newBall.run_hod(
            newBall.tracers,
            HOD_params['want_rsd'],
            write_to_disk=False,
            **hod_kwargs,
        )
        mock_dict = {tracer: mock_dict[tracer]}  # single tracer for CV and fit
    out['json']['ngal'] = int(len(mock_dict[tracer]['x']))
    out['json']['nbar'] = float(out['json']['ngal'] / newBall.lbox**3)

    measure_and_fit(
        newBall, mock_dict, config, stats, cv_pre, 'pre', timer, out, hod_kwargs
    )

    if want_recon:
        with timer('recon'):
            recon_dict = newBall.run_recon(
                mock_dict, recon_params, Nthread=config.get('Nthread', 16)
            )
        measure_and_fit(
            newBall, recon_dict, config, stats, cv_post, 'post', timer, out, hod_kwargs
        )

    out['json']['timing'] = timer.info
    np.savez(outfn.with_suffix('.npz'), **out['npz'])
    with open(outfn.with_suffix('.json'), 'w') as fp:
        json.dump(out['json'], fp, indent=2)
    print('Saved', outfn.with_suffix('.json'), flush=True)


def main(
    path2config,
    phase=None,
    simname=None,
    z=None,
    seeds=(1,),
    tracer='LRG',
    stats=('xi',),
    outdir=DEFAULTS['outdir'],
    no_recon=False,
    cv_pre='default',
    cv_post='default',
    nthread=16,
    overwrite=False,
    cv_workdir=None,
):
    with open(path2config) as fp:
        config = yaml.safe_load(fp)
    sim_params = config['sim_params']
    if simname is None and phase is not None:
        simname = f'AbacusSummit_base_c000_ph{phase:03d}'
    if simname is not None:
        sim_params['sim_name'] = simname
    if z is not None:
        sim_params['z_mock'] = z
    config['Nthread'] = nthread
    if any(seed < 1 for seed in seeds):
        raise ValueError('seeds must be positive integers (reseed=0 disables reseeding)')

    def none_if_raw(cv):
        return None if cv == 'none' else cv

    cv_pre, cv_post = none_if_raw(cv_pre), none_if_raw(cv_post)

    outdir = Path(os.path.expandvars(str(outdir)))
    outdir.mkdir(parents=True, exist_ok=True)

    workdirs = []
    if cv_workdir:
        # one private directory per process (the pid keeps concurrent processes apart)
        workdir = Path(os.path.expandvars(cv_workdir)) / (
            f'{sim_params["sim_name"]}_seeds{"-".join(map(str, seeds))}_{os.getpid()}'
        )
        workdirs = make_cv_workdir(config, workdir)
        print('Control-variate work directory', workdir, flush=True)
    try:
        run_seeds(config, sim_params, outdir, seeds, tracer, stats, no_recon, cv_pre, cv_post, overwrite)
    finally:
        if cv_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def run_seeds(config, sim_params, outdir, seeds, tracer, stats, no_recon, cv_pre, cv_post, overwrite):
    newBall = AbacusHOD(sim_params, config['HOD_params'], config['clustering_params'])
    for seed in seeds:
        outfn = (
            outdir
            / f'bao_{sim_params["sim_name"]}_z{sim_params["z_mock"]:.3f}_seed{seed}'
        )
        if outfn.with_suffix('.json').exists() and not overwrite:
            print('Exists, skipping', outfn, flush=True)
            continue
        print(f'=== {sim_params["sim_name"]} z={sim_params["z_mock"]} seed={seed}')
        run_seed(
            newBall,
            config,
            tracer,
            seed,
            stats,
            not no_recon,
            cv_pre,
            cv_post,
            outfn,
        )


class ArgParseFormatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=ArgParseFormatter
    )
    parser.add_argument(
        '--path2config', help='Path to the config file', default=DEFAULTS['path2config']
    )
    parser.add_argument('--phase', type=int, help='AbacusSummit_base_c000 phase number')
    parser.add_argument('--simname', help='Full simulation name (overrides --phase)')
    parser.add_argument('--z', type=float, help='Redshift (overrides the config)')
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=[1], help='HOD random seeds (>= 1)'
    )
    parser.add_argument('--tracer', default='LRG')
    parser.add_argument(
        '--stats', nargs='+', default=['xi'], choices=['xi', 'pk'], help='Fit xi and/or pk'
    )
    parser.add_argument('--outdir', default=DEFAULTS['outdir'])
    parser.add_argument('--no_recon', action='store_true', help='Only fit pre-recon')
    parser.add_argument(
        '--cv_pre',
        default='default',
        help="Control variates before recon: 'default' (ZCV), 'zcv' or 'none'",
    )
    parser.add_argument(
        '--cv_post',
        default='default',
        help="Control variates after recon: 'default' (LCV), 'lcv' or 'none'",
    )
    parser.add_argument('--nthread', type=int, default=16)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--cv_workdir',
        help='Run with a private symlinked copy of the control-variate inputs under this '
        'directory (removed at the end); required when several processes share a phase',
    )
    args = vars(parser.parse_args())
    main(**args)
