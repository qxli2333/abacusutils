"""
Generate HOD mocks, optionally run BAO reconstruction on them, measure the
power spectrum and correlation function multipoles with control variates
(linear control variates by default after reconstruction), and optionally fit the
BAO scale with desilike (``--fit_bao``, settings in ``bao_params``).

Requires pyrecon (``pip install git+https://github.com/cosmodesi/pyrecon@main``) and
the LCV inputs (see ``recon_params`` and ``lcv_params`` in the config file).

Usage
-----
$ python ./run_recon.py --help
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from abacusnbody.hod.abacus_hod import AbacusHOD

DEFAULTS = {}
DEFAULTS['path2config'] = 'config/abacus_hod.yaml'


def main(path2config, tracer='LRG', outfn=None, fit_bao=False):
    with open(path2config) as fp:
        config = yaml.safe_load(fp)
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    clustering_params = config['clustering_params']
    recon_params = config.get('recon_params') or {}

    newBall = AbacusHOD(sim_params, HOD_params, clustering_params)
    mock_dict = newBall.run_hod(
        newBall.tracers, HOD_params['want_rsd'], write_to_disk=False, Nthread=16
    )
    mock_dict = {tracer: mock_dict[tracer]}  # control variates use a single tracer

    if recon_params.get('want_recon', False):
        start = time.time()
        mock_dict = newBall.run_recon(mock_dict, recon_params, Nthread=16)
        print('Done recon, took time ', time.time() - start)

    # LCV after reconstruction, ZCV otherwise
    start = time.time()
    pk_dict = newBall.apply_cv(mock_dict, config, stat='pk')
    xi_dict = newBall.apply_cv(mock_dict, config, stat='xi')
    print('Done clustering, took time ', time.time() - start)

    bao = {}
    if fit_bao:  # requires desilike; settings from config['bao_params']
        stat = (config.get('bao_params') or {}).get('stat', 'xi')
        result = newBall.fit_bao(
            mock_dict, config, cv_dict=pk_dict if stat == 'pk' else xi_dict
        )
        for name in result['params']:
            print(f'{name} = {result[name]:.4f} +- {result[name + "_err"]:.4f}')
        print(f'chi2 / ndof = {result["chi2"]:.1f} / {result["ndof"]}')
        bao = {
            f'bao_{key}': np.asarray(result[key])
            for key in ['x', 'data', 'model', 'covariance', 'chi2', 'ndof']
        }
        for name in result['params']:
            bao[f'bao_{name}'] = result[name]
            bao[f'bao_{name}_err'] = result[name + '_err']

    if outfn is None:
        recon_str = '_recon' if recon_params.get('want_recon', False) else ''
        outfn = Path(sim_params['output_dir']) / f'clustering_{tracer}{recon_str}.npz'
    Path(outfn).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        outfn,
        **{key: np.asarray(val) for key, val in pk_dict.items()},
        **{key: np.asarray(val) for key, val in xi_dict.items() if key not in pk_dict},
        **bao,
    )
    print('Saved', outfn)


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
    parser.add_argument('--tracer', help='Tracer to measure', default='LRG')
    parser.add_argument('--outfn', help='Output .npz file')
    parser.add_argument(
        '--fit_bao', action='store_true', help='Fit the BAO scale with desilike'
    )
    args = vars(parser.parse_args())
    main(**args)
