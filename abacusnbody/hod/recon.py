r"""
Standard (Zel'dovich) BAO reconstruction of galaxy catalogs in a periodic box.

The reconstruction itself is delegated to an external engine; currently only
`pyrecon <https://github.com/cosmodesi/pyrecon>`_ is supported, which is not on
PyPI and must be installed from GitHub, e.g.::

    pip install git+https://github.com/cosmodesi/pyrecon@main

(the ``main`` branch is the OpenMP version; the default ``mpi`` branch also works but
requires ``pmesh``).

The reconstructed overdensity is :math:`\delta_{\rm rec} = \delta_D - \delta_S`, where
:math:`\delta_D` is the field of the shifted galaxies and :math:`\delta_S` is the field
of an initially uniform distribution shifted by the same (smoothed) displacement. In a
periodic box the unshifted distribution is exactly uniform, so by default
:math:`\delta_S` is sampled with a regular lattice rather than with random points,
which avoids the Poisson shot noise of randoms.

To linear order (Chen, Vlah & White 2019, arXiv:1907.00043):

* RecSym (galaxies and S catalog shifted by ``disp+rsd``):
  :math:`\delta_{\rm rec} = (b + f\mu^2)\,\delta_L`
* RecIso (galaxies shifted by ``disp+rsd``, S catalog by ``disp``):
  :math:`\delta_{\rm rec} = (b + f(1-\mathcal{S})\mu^2)\,\delta_L`, with
  :math:`\mathcal{S}(k) = e^{-k^2R^2/2}` the smoothing kernel.

These are the templates assumed by the linear control variates (LCV) in
:mod:`abacusnbody.hod.zcv.tools_cv`.
"""

import copy
import inspect

import numpy as np

__all__ = [
    'DEFAULT_RECON_BIAS',
    'DEFAULT_RECON_PARAMS',
    'get_recon_params',
    'make_lattice',
    'run_recon_pyrecon',
]

# fiducial linear biases used to estimate the displacement (DESI DR1 BAO choices)
DEFAULT_RECON_BIAS = {'LRG': 2.0, 'ELG': 1.2, 'QSO': 2.1}

DEFAULT_RECON_PARAMS = {
    'want_recon': False,
    'engine': 'pyrecon',
    'algorithm': 'IterativeFFTReconstruction',
    'convention': 'recsym',
    'smoothing_radius': 15.0,
    'nmesh': 512,
    'cellsize': None,
    'bias': DEFAULT_RECON_BIAS,
    'f': None,
    'los': 'z',
    'shifted_field': 'lattice',
    'lattice_nmesh': None,
    'nrandoms_factor': 10,
    'random_seed': 42,
    'recon_kwargs': {},
    'density_kwargs': {},
    'run_kwargs': {},
    'cv_type': 'lcv',
}

_ENGINES = ('pyrecon',)
_ALGORITHMS = (
    'IterativeFFTReconstruction',
    'MultiGridReconstruction',
    'IterativeFFTParticleReconstruction',
)
_CONVENTIONS = ('recsym', 'reciso')
_SHIFTED_FIELDS = ('lattice', 'randoms')


def get_recon_params(recon_params=None):
    """
    Merge user reconstruction settings over :data:`DEFAULT_RECON_PARAMS` and validate them.

    Parameters
    ----------
    recon_params : dict, optional
        User settings (e.g. the ``recon_params`` block of the config file).
        Unspecified keys take their default values.

    Returns
    -------
    params : dict
        Complete, validated settings.
    """
    params = copy.deepcopy(DEFAULT_RECON_PARAMS)
    user = dict(recon_params or {})
    unknown = set(user) - set(params)
    if unknown:
        raise ValueError(f'Unknown recon_params: {sorted(unknown)}')
    # an explicit cellsize takes precedence over the default nmesh
    if user.get('cellsize') is not None and 'nmesh' not in user:
        user['nmesh'] = None
    params.update(user)
    for key in ('recon_kwargs', 'density_kwargs', 'run_kwargs'):
        params[key] = dict(params[key] or {})

    if params['engine'] not in _ENGINES:
        raise NotImplementedError(
            f'Reconstruction engine {params["engine"]!r} not implemented; '
            f'available: {_ENGINES}'
        )
    if params['algorithm'] not in _ALGORITHMS:
        raise ValueError(
            f'Unknown algorithm {params["algorithm"]!r}; choose from {_ALGORITHMS}'
        )
    params['convention'] = params['convention'].lower()
    if params['convention'] not in _CONVENTIONS:
        raise ValueError(
            f'Unknown convention {params["convention"]!r}; choose from {_CONVENTIONS}'
        )
    if params['shifted_field'] not in _SHIFTED_FIELDS:
        raise ValueError(
            f'Unknown shifted_field {params["shifted_field"]!r}; '
            f'choose from {_SHIFTED_FIELDS}'
        )
    if (params['nmesh'] is None) == (params['cellsize'] is None):
        raise ValueError('Specify exactly one of `nmesh` or `cellsize` for recon.')
    if params['smoothing_radius'] <= 0:
        raise ValueError('`smoothing_radius` must be positive.')
    return params


def _get_bias(bias, tracer):
    """Bias for this tracer, from a float or a per-tracer dict."""
    if isinstance(bias, dict):
        if tracer not in bias:
            raise KeyError(
                f'No reconstruction bias given for tracer {tracer!r}; '
                f'add it to recon_params["bias"].'
            )
        return float(bias[tracer])
    return float(bias)


def _wrap_box(pos, Lbox, boxcenter=0.0):
    """Wrap positions into [boxcenter - L/2, boxcenter + L/2) as float32."""
    lo = boxcenter - Lbox / 2.0
    pos = ((np.asarray(pos, dtype=np.float64) - lo) % Lbox + lo).astype(np.float32)
    # float32 rounding can land exactly on the upper edge
    pos[pos >= np.float32(lo + Lbox)] = np.float32(lo)
    return pos


def make_lattice(n, Lbox, boxcenter=0.0, islab=None, dtype=np.float64):
    """
    Regular lattice of points at the cell centers of an ``n^3`` grid.

    Parameters
    ----------
    n : int
        number of lattice points per side.
    Lbox : float
        box size.
    boxcenter : float, optional
        center of the box; points lie in [boxcenter - L/2, boxcenter + L/2).
    islab : slice, optional
        range of x-indices to generate (to build the lattice slab by slab).
    dtype : np.dtype, optional
        output dtype.

    Returns
    -------
    pos : array_like
        lattice positions of shape (n_x * n * n, 3).
    """
    cell = Lbox / n
    coords = (np.arange(n) + 0.5) * cell + boxcenter - Lbox / 2.0
    xs = coords if islab is None else coords[islab]
    pos = np.empty((len(xs), n, n, 3), dtype=dtype)
    pos[..., 0] = xs[:, None, None]
    pos[..., 1] = coords[None, :, None]
    pos[..., 2] = coords[None, None, :]
    return pos.reshape(-1, 3)


def _uses_threads_api(pyrecon):
    """The pyrecon ``main`` branch is OpenMP-threaded; the ``mpi`` branch uses pmesh."""
    init = inspect.signature(pyrecon.recon.BaseReconstruction.__init__)
    return 'mpicomm' not in init.parameters


def run_recon_pyrecon(
    pos,
    Lbox,
    f,
    bias,
    algorithm='IterativeFFTReconstruction',
    convention='recsym',
    smoothing_radius=15.0,
    nmesh=None,
    cellsize=None,
    los='z',
    shifted_field='lattice',
    lattice_nmesh=None,
    nrandoms_factor=10,
    seed=42,
    nthread=16,
    boxcenter=0.0,
    recon_kwargs=None,
    density_kwargs=None,
    run_kwargs=None,
    max_slab_points=2**24,
):
    r"""
    Run reconstruction on a periodic-box catalog with pyrecon.

    Parameters
    ----------
    pos : array_like
        galaxy positions of shape (N, 3), in [boxcenter - L/2, boxcenter + L/2).
    Lbox : float
        box size in Mpc/h.
    f : float
        growth rate used to remove RSD (0 for real-space catalogs).
    bias : float
        linear bias used to convert the galaxy field into the displacement.
    algorithm : str, optional
        pyrecon class: ``'IterativeFFTReconstruction'`` (default),
        ``'MultiGridReconstruction'`` or ``'IterativeFFTParticleReconstruction'``.
    convention : str, optional
        ``'recsym'`` (default) or ``'reciso'``.
    smoothing_radius : float, optional
        Gaussian smoothing scale :math:`R` in Mpc/h, kernel :math:`e^{-k^2R^2/2}`.
    nmesh, cellsize : int, float, optional
        reconstruction mesh size or cell size (exactly one).
    los : str, optional
        line of sight axis. Default ``'z'``, as in AbacusHOD RSD.
    shifted_field : str, optional
        how to sample the shifted uniform field: ``'lattice'`` (default; regular
        grid, no shot noise) or ``'randoms'`` (uniform random points).
    lattice_nmesh : int, optional
        lattice points per side; default is the reconstruction mesh size.
        Matching the mesh used to measure the power spectrum is recommended,
        since then the unshifted lattice paints to exactly zero overdensity.
    nrandoms_factor : float, optional
        number of randoms per galaxy if ``shifted_field == 'randoms'``.
    seed : int, optional
        random seed if ``shifted_field == 'randoms'``.
    nthread : int, optional
        number of OpenMP threads (pyrecon ``main`` branch only).
    boxcenter : float, optional
        center of the box. Default 0, i.e. positions in [-L/2, L/2).
    recon_kwargs, density_kwargs, run_kwargs : dict, optional
        extra keyword arguments passed to the pyrecon constructor,
        ``set_density_contrast`` and ``run``, respectively.
    max_slab_points : int, optional
        maximum number of S-catalog points shifted at once (limits memory).

    Returns
    -------
    pos_rec : array_like
        reconstructed galaxy positions, shape (N, 3), float32.
    pos_shifted : array_like
        shifted lattice/randoms positions (the S catalog), shape (M, 3), float32.
    """
    try:
        import pyrecon
    except ImportError as e:
        raise ImportError(
            'Reconstruction requires pyrecon. Install it with '
            '"pip install git+https://github.com/cosmodesi/pyrecon@main".'
        ) from e

    convention = convention.lower()
    if convention not in _CONVENTIONS:
        raise ValueError(f'Unknown convention {convention!r}')
    if (nmesh is None) == (cellsize is None):
        raise ValueError('Specify exactly one of `nmesh` or `cellsize`.')
    recon_kwargs = dict(recon_kwargs or {})
    if _uses_threads_api(pyrecon):
        recon_kwargs.setdefault('nthreads', nthread)
    if nmesh is not None:
        recon_kwargs['nmesh'] = nmesh
    else:
        recon_kwargs['cellsize'] = cellsize

    Recon = getattr(pyrecon, algorithm)
    recon = Recon(
        f=f,
        bias=bias,
        los=los,
        boxsize=Lbox,
        boxcenter=boxcenter,
        wrap=True,
        **recon_kwargs,
    )
    pos = np.asarray(pos, dtype=np.float64)
    recon.assign_data(pos)
    # no randoms: the box is periodic and uniformly selected
    recon.set_density_contrast(
        smoothing_radius=smoothing_radius, **dict(density_kwargs or {})
    )
    recon.run(**dict(run_kwargs or {}))

    # galaxies are always shifted by the full (Zel'dovich + RSD) displacement
    if algorithm == 'IterativeFFTParticleReconstruction':
        pos_rec = recon.read_shifted_positions('data', field='disp+rsd')
    else:
        pos_rec = recon.read_shifted_positions(pos, field='disp+rsd')
    pos_rec = _wrap_box(pos_rec, Lbox, boxcenter)
    del pos

    # the S catalog: RecSym removes the large-scale RSD too, RecIso does not
    field = 'disp+rsd' if convention == 'recsym' else 'disp'
    if shifted_field == 'lattice':
        if lattice_nmesh is None:
            lattice_nmesh = (
                int(nmesh)
                if nmesh is not None
                else int(np.rint(Lbox / np.ravel(cellsize)[0]))
            )
        n = int(lattice_nmesh)
        nx_slab = max(1, int(max_slab_points // n**2))
        pos_shifted = np.empty((n**3, 3), dtype=np.float32)
        for i0 in range(0, n, nx_slab):
            islab = slice(i0, min(i0 + nx_slab, n))
            lattice = make_lattice(n, Lbox, boxcenter, islab=islab)
            shifted = recon.read_shifted_positions(lattice, field=field)
            pos_shifted[i0 * n**2 : islab.stop * n**2] = _wrap_box(
                shifted, Lbox, boxcenter
            )
            del lattice, shifted
    elif shifted_field == 'randoms':
        from parallel_numpy_rng import MTGenerator

        nrand = int(np.rint(nrandoms_factor * len(pos_rec)))
        mtg = MTGenerator(np.random.PCG64(seed))
        rand = mtg.random(size=3 * nrand, nthread=nthread, dtype=np.float64)
        rand = rand.reshape(nrand, 3) * Lbox + (boxcenter - Lbox / 2.0)
        pos_shifted = _wrap_box(
            recon.read_shifted_positions(rand, field=field), Lbox, boxcenter
        )
        del rand
    else:
        raise ValueError(f'Unknown shifted_field {shifted_field!r}')
    del recon

    return pos_rec, pos_shifted
