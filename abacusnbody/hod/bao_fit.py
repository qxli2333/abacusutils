r"""
BAO fits of HOD clustering measurements (power spectrum or correlation function
multipoles, before or after reconstruction, raw or with control variates) using
`desilike <https://github.com/cosmodesi/desilike>`_.

The default settings follow the DESI DR2 BAO baseline (DESI DR2 Results II and its
supporting papers):

* the BAO template of Chen et al. 2024 (desilike ``DampedBAOWigglesTracer*Multipoles``,
  ``model='standard'``), with the reconstruction convention and smoothing radius of the
  catalog (RecSym; 15 Mpc/h, or 30 Mpc/h for QSOs, by default);
* dilation parameters :math:`\alpha_{\rm iso}, \alpha_{\rm AP}` (``qiso``, ``qap``)
  with flat priors; flat priors on ``b1``, ``dbeta`` and :math:`\Sigma_s`;
* Gaussian priors on the BAO damping :math:`\Sigma_\parallel, \Sigma_\perp` (widths 2 and
  1 Mpc/h), with means depending on the tracer and on reconstruction;
* correlation function monopole and quadrupole in :math:`60 < s < 150` Mpc/h with
  4 Mpc/h bins, and the ``pcs2`` broadband (cubic splines in :math:`k` with spacing
  :math:`2\pi/r_d`, keeping the two lowest-order quadrupole terms, plus two terms per
  multipole for :math:`k < 0.02` h/Mpc), marginalized analytically;
* or the power spectrum monopole and quadrupole in :math:`0.02 < k < 0.3` h/Mpc with the
  ``pcs`` broadband.

The covariance is the Gaussian (disconnected) covariance of the periodic box, computed
from the measured multipoles including shot noise, or read from a file. When fitting
measurements with control variates, it is by default multiplied by the variance
reduction of the control variates.
"""

import copy
import re
import warnings

import numba
import numpy as np
from scipy.linalg import block_diag
from scipy.special import eval_legendre, spherical_jn

from .recon import DEFAULT_RECON_SMOOTHING_RADIUS

__all__ = [
    'DEFAULT_BAO_PARAMS',
    'DESI_SIGMA_PRIORS',
    'fit_bao',
    'gaussian_covariance_pk_poles',
    'gaussian_covariance_xi_poles',
    'get_bao_params',
    'get_fiducial_cosmology',
    'rebin_xi_poles',
]

# Means of the Gaussian priors on (Sigma_par, Sigma_per) in Mpc/h used by DESI DR1/DR2
# (DESI 2024 III, table 6), before and after reconstruction; the prior widths are
# DESI_SIGMA_PRIOR_WIDTHS. The QSO post-reconstruction values assume DESI's 30 Mpc/h
# smoothing radius.
DESI_SIGMA_PRIORS = {
    'pre': {
        'BGS': (10.0, 6.5),
        'LRG': (9.0, 4.5),
        'ELG': (8.5, 4.5),
        'QSO': (9.0, 3.5),
    },
    'post': {
        'BGS': (8.0, 3.0),
        'LRG': (6.0, 3.0),
        'ELG': (6.0, 3.0),
        'QSO': (6.0, 3.0),
    },
}
DESI_SIGMA_PRIOR_WIDTHS = (2.0, 1.0)

DEFAULT_BAO_PARAMS = {
    'stat': 'xi',
    'data': 'cv',
    'cv_type': 'default',
    'ells': (0, 2),
    'slim': (60.0, 150.0, 4.0),
    'klim': (0.02, 0.3),
    'apmode': 'qisoqap',
    'fiducial': 'simulation',
    'engine': 'class',
    'broadband': None,
    'model': 'standard',
    'mode': None,
    'smoothing_radius': None,
    'sigma_priors': None,
    'covariance': 'gaussian',
    'cov_rescale': 1.0,
    'cov_cv_reduction': True,
    'method': 'profile',
    'niterations': 3,
    'interval': False,
    'seed': 42,
    'sampler_kwargs': {},
    'run_kwargs': {},
}

_STATS = ('pk', 'xi')
_APMODES = ('qiso', 'qisoqap', 'qparqper')
_METHODS = ('profile', 'emcee')


def get_bao_params(bao_params=None):
    """
    Merge user BAO fit settings over :data:`DEFAULT_BAO_PARAMS` and validate them.

    Parameters
    ----------
    bao_params : dict, optional
        User settings (e.g. the ``bao_params`` block of the config file).

    Returns
    -------
    params : dict
        Complete, validated settings.
    """
    params = copy.deepcopy(DEFAULT_BAO_PARAMS)
    user = dict(bao_params or {})
    unknown = set(user) - set(params)
    if unknown:
        raise ValueError(f'Unknown bao_params: {sorted(unknown)}')
    params.update(user)
    if params['stat'] not in _STATS:
        raise ValueError(f'stat should be one of {_STATS}')
    if params['apmode'] not in _APMODES:
        raise ValueError(f'apmode should be one of {_APMODES}')
    if params['method'] not in _METHODS:
        raise ValueError(f'method should be one of {_METHODS}')
    if params['data'] not in ('cv', 'raw'):
        raise ValueError("data should be 'cv' or 'raw'")
    params['ells'] = tuple(int(ell) for ell in params['ells'])
    if any(ell not in (0, 2, 4) for ell in params['ells']):
        raise ValueError('Only multipoles 0, 2, 4 are supported')
    if params['broadband'] is None:
        params['broadband'] = 'pcs2' if params['stat'] == 'xi' else 'pcs'
    for key in ('sampler_kwargs', 'run_kwargs'):
        params[key] = dict(params[key] or {})
    return params


def get_fiducial_cosmology(fiducial='simulation', sim_name=None, engine='class'):
    """
    Fiducial cosmology of the BAO template, as a :class:`cosmoprimo.Cosmology`.

    Parameters
    ----------
    fiducial : str or cosmoprimo.Cosmology, optional
        ``'simulation'`` (default) for the cosmology of the AbacusSummit simulation
        ``sim_name`` (so that the expected dilation parameters are 1), the name of a
        cosmology in :mod:`cosmoprimo.fiducial` (e.g. ``'DESI'``), or a cosmology instance.
    sim_name : str, optional
        simulation name, e.g. ``'AbacusSummit_base_c000_ph006'``.
    engine : str, optional
        cosmoprimo engine for the linear power spectrum. Default ``'class'``.
    """
    from cosmoprimo import fiducial as cosmoprimo_fiducial

    if not isinstance(fiducial, str):
        return fiducial
    if fiducial == 'simulation':
        match = re.search(r'_c(\d{3})_', str(sim_name))
        if match is None:
            raise ValueError(
                f'Cannot infer the AbacusSummit cosmology of {sim_name!r}; '
                "set bao_params['fiducial'] (e.g. 'DESI')."
            )
        return cosmoprimo_fiducial.AbacusSummit(name=match.group(1), engine=engine)
    return getattr(cosmoprimo_fiducial, fiducial)(engine=engine)


def _get_variance_ratio(variance_ratio, poles, ell, nk):
    """Variance reduction of the control variates for multipole ``ell`` (1 if unknown)."""
    if variance_ratio is None:
        return np.ones(nk)
    variance_ratio = np.atleast_2d(variance_ratio)
    if variance_ratio.shape[0] == 1:
        return variance_ratio[0]
    return variance_ratio[list(poles).index(ell)]


def _mu_average_pk2(pk_ell, poles, ell1, ell2, nmu=40):
    r""":math:`\frac{1}{2}\int_{-1}^{1} d\mu\, P^2(k, \mu) L_{\ell_1}(\mu) L_{\ell_2}(\mu)`."""
    mu, weights = np.polynomial.legendre.leggauss(nmu)
    pkmu = sum(
        np.asarray(pk)[:, None] * eval_legendre(ell, mu)[None, :]
        for ell, pk in zip(poles, pk_ell)
    )
    integrand = pkmu**2 * eval_legendre(ell1, mu) * eval_legendre(ell2, mu)
    return 0.5 * np.sum(weights * integrand, axis=-1)


def _gaussian_variance(pk_ell, nmodes, poles, ell1, ell2, variance_ratio=None):
    r"""
    Per-mode-bin variance term
    :math:`2 (2\ell_1+1)(2\ell_2+1) \langle P^2 L_{\ell_1} L_{\ell_2} \rangle_\mu / N_k`
    (``nmodes`` counts the modes of both hemispheres), optionally times the control
    variate variance reduction.
    """
    nmodes = np.asarray(nmodes, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma2 = (
            2.0
            * (2 * ell1 + 1)
            * (2 * ell2 + 1)
            * _mu_average_pk2(pk_ell, poles, ell1, ell2)
            / nmodes
        )
    sigma2[~np.isfinite(sigma2)] = 0.0
    if variance_ratio is not None:
        nk = len(nmodes)
        sigma2 *= np.sqrt(
            _get_variance_ratio(variance_ratio, poles, ell1, nk)
            * _get_variance_ratio(variance_ratio, poles, ell2, nk)
        )
    return sigma2


def gaussian_covariance_pk_poles(
    pk_ell, nmodes, poles, ells=(0, 2), variance_ratio=None
):
    r"""
    Gaussian covariance of the power spectrum multipoles measured in a periodic box,

    .. math:: C_{\ell_1 \ell_2}(k_i, k_j) = \delta_{ij} \frac{2(2\ell_1+1)(2\ell_2+1)}{N_i}
              \langle P^2(k_i, \mu) L_{\ell_1}(\mu) L_{\ell_2}(\mu) \rangle_\mu,

    with :math:`P(k, \mu) = \sum_\ell P_\ell(k) L_\ell(\mu)` including shot noise.

    Parameters
    ----------
    pk_ell : array_like
        measured (or smooth) multipoles, shape (len(poles), nk), including shot noise.
    nmodes : array_like
        number of modes per k bin (both hemispheres), shape (nk,).
    poles : tuple
        multipoles of ``pk_ell``.
    ells : tuple, optional
        multipoles of the output covariance.
    variance_ratio : array_like, optional
        variance reduction of control variates, shape (len(poles), nk) or (1, nk).

    Returns
    -------
    cov : array_like
        covariance of shape (len(ells) * nk, len(ells) * nk), ordered by multipole.
    """
    nk = len(nmodes)
    cov = np.zeros((len(ells) * nk, len(ells) * nk))
    for i1, ell1 in enumerate(ells):
        for i2, ell2 in enumerate(ells):
            sigma2 = _gaussian_variance(
                pk_ell, nmodes, poles, ell1, ell2, variance_ratio=variance_ratio
            )
            cov[i1 * nk : (i1 + 1) * nk, i2 * nk : (i2 + 1) * nk] = np.diag(sigma2)
    return cov


def _bin_averaged_jl(ell, k, s_edges, nsub=16):
    r"""Average of :math:`j_\ell(ks)` over each ``s`` bin, weighted by :math:`s^2`."""
    s_edges = np.asarray(s_edges, dtype=float)
    frac = (np.arange(nsub) + 0.5) / nsub
    s = s_edges[:-1, None] + frac[None, :] * np.diff(s_edges)[:, None]  # (ns, nsub)
    weights = s**2 / np.sum(s**2, axis=-1, keepdims=True)
    jl = spherical_jn(ell, np.asarray(k)[None, None, :] * s[:, :, None])
    return np.sum(weights[:, :, None] * jl, axis=1)  # (ns, nk)


@numba.njit(cache=True)
def _lattice_mu2_moments(n1d, Lbox, dk, nbins, nmom):
    r"""
    Sums over all Fourier modes of an ``n1d^3`` mesh (both hemispheres) of
    :math:`\mu^{2m}`, :math:`m = 0, \ldots, nmom - 1`, in bins of :math:`|k|` of width ``dk``.
    """
    kf = 2.0 * np.pi / Lbox
    moments = np.zeros((nbins, nmom))
    kzlen = n1d // 2 + 1
    for i in range(n1d):
        ix = i if i < n1d // 2 else i - n1d
        for j in range(n1d):
            iy = j if j < n1d // 2 else j - n1d
            for iz in range(kzlen):
                n2 = ix * ix + iy * iy + iz * iz
                if n2 == 0:
                    continue
                # modes with 0 < kz < k_Ny stand for both +k and -k
                weight = 1.0 if (iz == 0 or (n1d % 2 == 0 and iz == n1d // 2)) else 2.0
                ibin = int(kf * np.sqrt(n2) / dk)
                if ibin >= nbins:
                    continue
                mu2 = iz * iz / n2
                power = 1.0
                for m in range(nmom):
                    moments[ibin, m] += weight * power
                    power *= mu2
    return moments


# Legendre polynomials L_0, L_2, L_4 as polynomials in x = mu^2 (increasing powers)
_LEGENDRE_MU2 = {
    0: np.array([1.0]),
    2: np.array([-0.5, 1.5]),
    4: np.array([3.0 / 8.0, -30.0 / 8.0, 35.0 / 8.0]),
}


def gaussian_covariance_xi_poles(
    s_edges,
    k,
    pk_ell,
    poles,
    Lbox,
    nmesh,
    ells=(0, 2),
    variance_ratio=None,
    oversampling=8,
):
    r"""
    Gaussian covariance of the correlation function multipoles measured in a periodic
    box of volume :math:`V` by Fourier transforming the 3D power spectrum on a mesh,

    .. math:: C_{\ell_1 \ell_2}(s_i, s_j) = \frac{2(2\ell_1+1)(2\ell_2+1)(-1)^{(\ell_1-\ell_2)/2}}{V^2}
              \sum_{\mathbf{k}} P^2(\mathbf{k}) L_{\ell_1}(\mu) L_{\ell_2}(\mu)
              \bar{j}_{\ell_1}(k s_i) \bar{j}_{\ell_2}(k s_j),

    where the sum runs over all the modes of the mesh (including beyond the Nyquist
    frequency), :math:`P(k, \mu) = \sum_\ell P_\ell(k) L_\ell(\mu)` and :math:`\bar{j}_\ell`
    are spherical Bessel functions averaged over the ``s`` bins. The sum over modes is
    done exactly in thin :math:`|k|` shells (width :math:`k_f` / ``oversampling``), using
    the :math:`\mu` moments of the modes of each shell.

    Parameters
    ----------
    s_edges : array_like
        edges of the separation bins, in Mpc/h.
    k : array_like
        k bin centers of the measured multipoles, up to (about) the Nyquist frequency;
        multipoles are interpolated between, and held constant beyond, them.
    pk_ell : array_like
        power spectrum multipoles including shot noise, shape (len(poles), nk).
    poles : tuple
        multipoles of ``pk_ell``.
    Lbox : float
        box size in Mpc/h.
    nmesh : int
        size of the mesh on which the correlation function was measured.
    ells : tuple, optional
        multipoles of the output covariance.
    variance_ratio : array_like, optional
        variance reduction of control variates, shape (len(poles), nk) or (1, nk).
    oversampling : int, optional
        number of thin shells per fundamental-mode width.

    Returns
    -------
    cov : array_like
        covariance of shape (len(ells) * ns, len(ells) * ns), ordered by multipole.
    """
    volume = float(Lbox) ** 3
    k = np.asarray(k, dtype=float)
    pk_ell = np.asarray(pk_ell, dtype=float).reshape(len(poles), len(k))
    ns = len(s_edges) - 1

    # mu^2 moments of the modes in thin shells, up to the corners of the mesh
    dk = 2.0 * np.pi / float(Lbox) / oversampling
    kmax = np.sqrt(3.0) * np.pi * nmesh / float(Lbox)
    nbins = int(kmax / dk) + 2
    degree = max(poles) + max(ells)  # of P^2 L_ell1 L_ell2 as a polynomial in mu^2
    moments = _lattice_mu2_moments(int(nmesh), float(Lbox), dk, nbins, degree + 1)
    nonzero = moments[:, 0] > 0
    moments = moments[nonzero]
    kshell = (np.arange(nbins)[nonzero] + 0.5) * dk

    # P(k, mu) as a polynomial in mu^2 in each shell
    ppoly = np.zeros((len(kshell), max(poles) // 2 + 1))
    for ell, pk in zip(poles, pk_ell):
        coeffs = _LEGENDRE_MU2[ell]
        ppoly[:, : len(coeffs)] += np.interp(kshell, k, pk)[:, None] * coeffs[None, :]
    p2poly = np.array([np.convolve(c, c) for c in ppoly])

    jbar = {ell: _bin_averaged_jl(ell, kshell, s_edges) for ell in ells}
    cov = np.zeros((len(ells) * ns, len(ells) * ns))
    for i1, ell1 in enumerate(ells):
        for i2, ell2 in enumerate(ells):
            lpoly = np.convolve(_LEGENDRE_MU2[ell1], _LEGENDRE_MU2[ell2])
            fpoly = np.array([np.convolve(c, lpoly) for c in p2poly])
            # sum over the modes of each shell of P^2 L_ell1 L_ell2
            summed = np.sum(fpoly * moments[:, : fpoly.shape[1]], axis=-1)
            if variance_ratio is not None:
                summed = summed * np.sqrt(
                    np.interp(
                        kshell,
                        k,
                        _get_variance_ratio(variance_ratio, poles, ell1, len(k)),
                    )
                    * np.interp(
                        kshell,
                        k,
                        _get_variance_ratio(variance_ratio, poles, ell2, len(k)),
                    )
                )
            norm = (
                2.0 * (2 * ell1 + 1) * (2 * ell2 + 1) * (-1) ** ((ell1 - ell2) // 2)
            ) / volume**2
            block = (jbar[ell1] * (norm * summed)[None, :]) @ jbar[ell2].T
            cov[i1 * ns : (i1 + 1) * ns, i2 * ns : (i2 + 1) * ns] = block
    return cov


def _rebin_matrix(r_binc, counts, s_edges):
    """
    Matrix averaging fine separation bins into coarser bins, weighted by counts.

    Returns
    -------
    r_in : array_like
        centers of the input bins used, shape (nin,).
    matrix : array_like
        rebinning matrix, shape (len(s_edges) - 1, nin).
    """
    r_binc = np.asarray(r_binc, dtype=float)
    counts = np.asarray(counts, dtype=float).reshape(-1)[: len(r_binc)]
    s_edges = np.asarray(s_edges, dtype=float)
    ibin = np.digitize(r_binc, s_edges) - 1
    used = (ibin >= 0) & (ibin < len(s_edges) - 1) & (counts > 0)
    matrix = np.zeros((len(s_edges) - 1, used.sum()))
    for i in range(len(s_edges) - 1):
        mask = ibin[used] == i
        if not mask.any():
            raise ValueError(f'No input bin in [{s_edges[i]}, {s_edges[i + 1]})')
        matrix[i, mask] = counts[used][mask] / counts[used][mask].sum()
    return r_binc[used], matrix


def rebin_xi_poles(r_binc, xi_ell, counts, s_edges):
    """
    Rebin correlation function multipoles to coarser separation bins, weighting the
    input bins by their number of pairs (mesh cells).

    Parameters
    ----------
    r_binc : array_like
        input bin centers, shape (nr,).
    xi_ell : array_like
        input multipoles, shape (npoles, nr).
    counts : array_like
        number of mesh cells per input bin, shape (nr,).
    s_edges : array_like
        output bin edges; each must coincide with an input bin edge.

    Returns
    -------
    s_mid : array_like
        weighted mean separation of each output bin.
    xi_rebinned : array_like
        rebinned multipoles, shape (npoles, len(s_edges) - 1).
    """
    r_binc = np.asarray(r_binc, dtype=float)
    xi_ell = np.atleast_2d(xi_ell)
    counts = np.asarray(counts, dtype=float).reshape(-1)[: len(r_binc)]
    r_in, matrix = _rebin_matrix(r_binc, counts, s_edges)
    used = np.isin(r_binc, r_in)
    return matrix @ r_in, xi_ell[:, used] @ matrix.T


def _get_key(cv_dict, base, data):
    """Key of the raw or control-variate-reduced measurement in ``cv_dict``."""
    if data == 'cv':
        for suffix in ('_lcv', '_zcv'):
            if base + suffix in cv_dict:
                return base + suffix
        warnings.warn(f'No control variates found for {base}; fitting the raw data.')
    return base


def _get_sigma_priors(sigma_priors, tracer, recon):
    """Gaussian priors (loc, scale) on sigmapar and sigmaper."""
    if sigma_priors is not None:
        return {name: tuple(sigma_priors[name]) for name in ('sigmapar', 'sigmaper')}
    table = DESI_SIGMA_PRIORS['post' if recon else 'pre']
    if tracer not in table:
        raise KeyError(
            f'No default BAO damping priors for tracer {tracer!r}; set '
            "bao_params['sigma_priors'] = {'sigmapar': [loc, scale], 'sigmaper': [loc, scale]}."
        )
    sigmapar, sigmaper = table[tracer]
    return {
        'sigmapar': (sigmapar, DESI_SIGMA_PRIOR_WIDTHS[0]),
        'sigmaper': (sigmaper, DESI_SIGMA_PRIOR_WIDTHS[1]),
    }


def fit_bao(
    cv_dict,
    z,
    Lbox,
    nbar=None,
    tracer='LRG',
    recon_info=None,
    sim_name=None,
    nmesh=None,
    bao_params=None,
):
    r"""
    Fit the BAO scale in power spectrum or correlation function multipoles with desilike.

    Parameters
    ----------
    cv_dict : dict
        measurement, output of ``AbacusHOD.apply_cv`` (``apply_lcv``, ``apply_lcv_xi``,
        ``apply_zcv``, ``apply_zcv_xi`` or the raw measurement with ``cv_type=None``)
        for ``bao_params['stat']``.
    z : float
        redshift of the measurement.
    Lbox : float
        box size in Mpc/h.
    nbar : float, optional
        number density in (h/Mpc)^3; its inverse (Poisson shot noise) is subtracted from
        the power spectrum monopole. Required for ``stat='pk'``.
    tracer : str, optional
        tracer name, used to choose the default damping priors.
    recon_info : dict, optional
        reconstruction settings (``recon_dict[tracer]['recon_info']``); ``None`` for
        pre-reconstruction measurements.
    sim_name : str, optional
        simulation name, used for ``fiducial='simulation'``.
    nmesh : int, optional
        mesh size of the correlation function measurement (for its Gaussian covariance);
        by default inferred from the k bins, assumed linear up to the Nyquist frequency.
    bao_params : dict, optional
        fit settings, see :data:`DEFAULT_BAO_PARAMS`.

    Returns
    -------
    result : dict
        best fit (or posterior mean) and error of the BAO parameters (e.g. ``'qiso'``,
        ``'qiso_err'``), ``'bestfit'`` and ``'error'`` dicts for all parameters, ``'chi2'``
        and ``'ndof'``, the fitted data vector, covariance and best-fit model, and the
        desilike ``'profiles'`` or ``'chain'``.
    """
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.theories.galaxy_clustering import BAOPowerSpectrumTemplate

    params = get_bao_params(bao_params)
    stat, ells = params['stat'], params['ells']
    recon = recon_info is not None

    # reconstruction mode of the template
    mode = params['mode']
    if mode is None:
        mode = recon_info['convention'] if recon else ''
    smoothing_radius = params['smoothing_radius']
    if smoothing_radius is None:
        if recon:
            smoothing_radius = recon_info['smoothing_radius']
        else:
            smoothing_radius = DEFAULT_RECON_SMOOTHING_RADIUS.get(tracer, 15.0)

    # data vector and covariance
    poles = tuple(int(ell) for ell in cv_dict['poles'])
    missing = set(ells) - set(poles)
    if missing:
        raise ValueError(f'Multipoles {sorted(missing)} not in the measurement')
    pk_key = _get_key(cv_dict, 'Pk_tr_tr_ell', params['data'])
    is_cv = pk_key != 'Pk_tr_tr_ell'
    variance_ratio = None
    if is_cv and params['cov_cv_reduction']:
        variance_ratio = cv_dict.get('cv_variance_ratio')
        if variance_ratio is None:
            warnings.warn('No cv_variance_ratio in the measurement; not reducing cov.')
    k_all = np.asarray(cv_dict['k_binc'])
    pk_ell = np.asarray(cv_dict[pk_key]).reshape(len(poles), len(k_all))
    nmodes = np.asarray(cv_dict['Nk_tr_tr_ell']).reshape(-1)[: len(k_all)]

    if stat == 'pk':
        if nbar is None:
            raise ValueError('nbar is required to subtract shot noise for stat="pk"')
        mask = (k_all >= params['klim'][0]) & (k_all <= params['klim'][1])
        x = k_all[mask]
        data = pk_ell[[poles.index(ell) for ell in ells]][:, mask].copy()
        if 0 in ells:
            data[ells.index(0)] -= 1.0 / nbar  # Poisson shot noise
        if params['covariance'] == 'gaussian':
            cov = gaussian_covariance_pk_poles(
                pk_ell, nmodes, poles, ells=ells, variance_ratio=variance_ratio
            )
            imask = np.concatenate([mask] * len(ells))
            cov = cov[np.ix_(imask, imask)]
    else:
        xi_key = _get_key(cv_dict, 'Xi_tr_tr_ell', params['data'])
        smin, smax, ds = params['slim']
        s_edges = np.arange(smin, smax + ds / 2.0, ds)
        s_edges = s_edges[s_edges <= smax + 1e-8]
        xi_ell = np.asarray(cv_dict[xi_key]).reshape(len(poles), -1)
        x, data = rebin_xi_poles(
            cv_dict['r_binc'], xi_ell, cv_dict['Np_tr_tr_ell'], s_edges
        )
        data = data[[poles.index(ell) for ell in ells]]
        # the theory is averaged over the separation bins as the data
        s_in, rebin = _rebin_matrix(cv_dict['r_binc'], cv_dict['Np_tr_tr_ell'], s_edges)
        if params['covariance'] == 'gaussian':
            if nmesh is None:  # k bins are linear up to the Nyquist frequency
                kf = 2.0 * np.pi / Lbox
                nmesh = int(np.rint(2.0 * (k_all[-1] + 0.5 * kf) / kf))
            if recon and recon_info.get('nmesh') and Lbox / recon_info['nmesh'] > ds:
                warnings.warn(
                    f'The reconstruction mesh cell size ({Lbox / recon_info["nmesh"]:.1f} '
                    f'Mpc/h) is larger than the separation bins ({ds} Mpc/h): the '
                    'interpolation of the displacement imprints the reconstruction mesh '
                    'period on the correlation function; use a finer reconstruction mesh.'
                )
            if Lbox / nmesh > ds:
                warnings.warn(
                    f'The mesh cell size ({Lbox / nmesh:.1f} Mpc/h) is larger than the '
                    f'separation bins ({ds} Mpc/h): the Gaussian covariance does not capture '
                    'the discreteness of the mesh correlation function estimator, use a '
                    'finer mesh or a covariance file.'
                )
            cov = gaussian_covariance_xi_poles(
                s_edges,
                k_all,
                pk_ell,
                poles,
                Lbox,
                nmesh,
                ells=ells,
                variance_ratio=variance_ratio,
            )
    if params['covariance'] != 'gaussian':
        fn = str(params['covariance'])
        cov = np.load(fn) if fn.endswith('.npy') else np.loadtxt(fn)
        if cov.shape != (data.size, data.size):
            raise ValueError(
                f'Covariance {fn} has shape {cov.shape}, but the data vector '
                f'has size {data.size} (multipoles {ells}, {len(x)} bins each)'
            )
        if variance_ratio is not None:
            warnings.warn(
                'The control variate variance reduction is not applied to a covariance '
                'read from a file.'
            )
    cov = cov * params['cov_rescale']
    if np.linalg.eigvalsh(cov).min() <= 0.0:
        raise ValueError(
            'The covariance is not positive definite; for the correlation function, '
            'measure it on a mesh with cells smaller than the separation bins, or '
            'provide a covariance file.'
        )
    flatdata = data.ravel()

    # theory and likelihood
    cosmo = get_fiducial_cosmology(params['fiducial'], sim_name, params['engine'])
    template = BAOPowerSpectrumTemplate(z=z, fiducial=cosmo, apmode=params['apmode'])
    theory_kwargs = {
        'template': template,
        'mode': mode,
        'smoothing_radius': smoothing_radius,
        'ells': ells,
        'broadband': params['broadband'],
        'model': params['model'],
    }
    if stat == 'pk':
        from desilike.observables.galaxy_clustering import (
            TracerPowerSpectrumMultipolesObservable,
        )
        from desilike.theories.galaxy_clustering import (
            DampedBAOWigglesTracerPowerSpectrumMultipoles,
        )

        theory = DampedBAOWigglesTracerPowerSpectrumMultipoles(k=x, **theory_kwargs)
        observable = TracerPowerSpectrumMultipolesObservable(
            data=flatdata, covariance=cov, ells=ells, k=x, theory=theory
        )
    else:
        from desilike.observables.galaxy_clustering import (
            TracerCorrelationFunctionMultipolesObservable,
        )
        from desilike.theories.galaxy_clustering import (
            DampedBAOWigglesTracerCorrelationFunctionMultipoles,
        )

        theory = DampedBAOWigglesTracerCorrelationFunctionMultipoles(**theory_kwargs)
        observable = TracerCorrelationFunctionMultipolesObservable(
            data=flatdata,
            covariance=cov,
            ells=ells,
            s=x,
            wmatrix=block_diag(*[rebin] * len(ells)).T,
            sin=s_in,
            theory=theory,
        )
    likelihood = ObservablesGaussianLikelihood(observables=[observable])

    # priors: Gaussian on the BAO damping
    for name, (loc, scale) in _get_sigma_priors(
        params['sigma_priors'], tracer, recon
    ).items():
        for param in likelihood.all_params.select(basename=name):
            param.update(
                fixed=False,
                value=loc,
                prior={'dist': 'norm', 'loc': loc, 'scale': scale},
                ref={'dist': 'norm', 'loc': loc, 'scale': scale / 4.0},
            )
    likelihood()  # initialize, so that the theory keeps only the relevant broadband
    broadband = [
        param.name
        for param in likelihood.all_params.select(basename=['al*_*', 'bl*_*'])
        if not param.fixed
    ]

    # The broadband terms are linear in the model, with a design matrix that does not
    # depend on the other parameters. For sampling, they are marginalized over
    # analytically (flat priors) by projecting them out of the precision matrix.
    model0 = np.asarray(observable.flattheory)
    design = []
    for name in broadband:
        likelihood(**{name: 1.0})
        design.append(np.asarray(observable.flattheory) - model0)
        likelihood(**{name: 0.0})
    design = np.array(design)
    precision = np.linalg.inv(cov)

    def solve_broadband(model_nobb):
        """Best-fit broadband amplitudes given the model without broadband."""
        pt = design @ precision
        return np.linalg.solve(pt @ design.T, pt @ (flatdata - model_nobb))

    main_params = {
        'qiso': ['qiso'],
        'qisoqap': ['qiso', 'qap'],
        'qparqper': ['qpar', 'qper'],
    }[params['apmode']]
    result = {
        'stat': stat,
        'tracer': tracer,
        'z': z,
        'mode': mode,
        'smoothing_radius': smoothing_radius,
        'ells': ells,
        'x': x,
        'data': data,
        'covariance': cov,
        'params': main_params,
    }
    if params['method'] == 'profile':
        from desilike.profilers import MinuitProfiler

        # the broadband is solved for analytically at each step
        for param in likelihood.all_params.select(name=broadband):
            param.update(derived='.auto')
        profiler = MinuitProfiler(likelihood, seed=params['seed'])
        profiles = profiler.maximize(niterations=params['niterations'])
        if params['interval']:
            profiles = profiler.interval(params=main_params)
        best = profiles.choice(index='argmax')  # best of the Minuit starts
        bestfit = best.bestfit.choice(input=True)
        result['bestfit'] = {
            name: float(np.ravel(value)[0]) for name, value in bestfit.items()
        }
        result['error'] = {
            name: float(np.ravel(best.error[name])[0])
            for name in best.error.names()
            if name in bestfit
        }
        result['profiles'] = profiles
        varied = [name for name in bestfit if name not in broadband]
    else:
        from desilike.samplers import EmceeSampler

        for param in likelihood.all_params.select(name=broadband):
            param.update(fixed=True, value=0.0)
        pt = design @ precision
        projected = precision - pt.T @ np.linalg.solve(pt @ design.T, pt)
        likelihood = ObservablesGaussianLikelihood(
            observables=[observable], precision=projected
        )
        varied = likelihood.varied_params.names()
        sampler_kwargs = {
            'n_walkers': 4 * len(varied),
            'rng': params['seed'],
            **params['sampler_kwargs'],
        }
        sampler = EmceeSampler(likelihood, **sampler_kwargs)
        # Gelman-Rubin is not valid for ensemble samplers: use the effective sample size
        run_kwargs = {
            'burn_in': 0.3,
            'gelman_rubin': None,
            'ess': 50,
            'max_steps': 20000,
            **params['run_kwargs'],
        }
        chain = sampler.run(**run_kwargs)
        result['bestfit'] = {name: float(chain.mean(name)) for name in varied}
        result['error'] = {name: float(chain.std(name)) for name in varied}
        result['chain'] = chain
        bestfit = chain.choice(index='argmax', params=varied)
        bestfit = {name: float(np.ravel(value)[0]) for name, value in bestfit.items()}

    # goodness of fit at the best fit, with the best-fit broadband
    nobb = {**{name: bestfit[name] for name in varied}, **dict.fromkeys(broadband, 0.0)}
    likelihood(
        **{name: value for name, value in nobb.items() if name in likelihood.all_params}
    )
    model_nobb = np.asarray(observable.flattheory)
    if len(broadband):
        model = model_nobb + solve_broadband(model_nobb) @ design
    else:
        model = model_nobb
    diff = flatdata - model
    result['model'] = model.reshape(data.shape)
    result['chi2'] = float(diff @ precision @ diff)
    result['ndof'] = flatdata.size - len(varied) - len(broadband)
    for name in main_params:
        result[name] = result['bestfit'][name]
        result[name + '_err'] = result['error'].get(name, np.nan)
    return result
