"""
Functions for doing multidimensional scaling.
"""
import logging
from typing import Callable, Optional

import numpy as np
import scipy.optimize as opt
import scipy.spatial.distance as dist
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


def make_edm(coords: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Make an Euclidean distance matrix from a set of points.

    Parameters
    ----------
    coords : NDArray[np.floating]
        The (npoint, ndim) array of input points.

    Returns
    -------
    edm : NDArray[np.floating]
        The (npoint, npoint) euclidean distance matrix.
    """
    dist_vec = dist.pdist(coords)
    edm = dist.squareform(dist_vec)

    return edm


def classic_mds(
    distance_matrix: NDArray[np.floating], ndim: int = 3
) -> NDArray[np.floating]:
    """
    Perform classical (Torgerson) MDS. This assumes a complete euclidean distance matrix.

    Parameters
    ----------
    distance_matrix : NDArray[np.floating]
        The euclidean distance matrix.
        Should be (npoint, npoint) and complete (no missing distances).
    ndim : int, default: 3
        The number of dimensions to scale to.

    Returns
    -------
    coords : NDArray[np.floating]
        The output coordinates, will be (npoint, ndim).
        Points are in the same order as distance_matrix.

    Raises
    ------
    ValueError
        If distance_matrix is not square.
        If distance_matrix has non-finite values.
    """
    npoint = len(distance_matrix)
    if distance_matrix.shape != (npoint, npoint):
        raise ValueError("Distance matrix should be square")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("Distance matrix must only have finite values")

    d_sq = distance_matrix**2
    cent_mat = np.eye(npoint) - np.ones_like(distance_matrix) / npoint
    double_cent = -0.5 * cent_mat @ d_sq @ cent_mat

    eigen_vals, eigen_vecs = np.linalg.eig(double_cent)
    eigen_vals = eigen_vals[-1 * ndim :]
    eigen_vecs = eigen_vecs[:, -1 * ndim :]
    tol = 1e16
    eigen_vals[(eigen_vals < 0) & (eigen_vals > -tol)] = 0.0

    coords = eigen_vecs @ (np.diag(np.sqrt(eigen_vals)))
    return np.real(coords)


def _init_coords(
    npoint: int, ndim: int, distance_matrix: Optional[NDArray[np.floating]] = None
) -> NDArray[np.floating]:
    if distance_matrix is None:
        scale = 1
    else:
        scale = np.nanmedian(distance_matrix)
    coords = np.random.default_rng().normal(0, scale, (npoint, ndim))
    return coords


def smacof(
    stress_func: Callable[
        [NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], int], float
    ],
    coords: NDArray[np.floating],
    distance_matrix: NDArray[np.floating],
    weights: NDArray[np.floating],
    ndim: int,
    max_iters: int = 10000,
    epsilon: float = 1e-10,
    verbose: bool = True,
) -> tuple[NDArray[np.floating], float]:
    """
    SMACOF algorithm for multidimensional scaling.

    Parameters
    ----------
    stress_func : Callable[[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], int], float]
        The stress function to use.
    coords : NDArray[np.floating]
        Initial guess of coordinates to calculate stress at.
        Should be (npoint, ndim)
    distance_matrix : NDArray[np.floating]
        The distance matrix.
        Should be (npoint, npoint), unknown distances should be set to nan.
    weights : NDArray[np.floating]
        How much to weigh each distance in distance_matrix in the metric.
        Should be (npoint, npoint) and have 1-to-1 correspondance with distance_matrix.
    ndim : int, default: 3
        The number of dimensions to scale to.
    epsilon : float, default: 1e-10
        The difference in stress between iterations before stopping.
    max_iters : int, default: 10000
        The maximum iterations to run for.
    verbose : bool, default: True
        Sets the verbosity of this function.
        If True logs at the INFO level.
        If False logs at the DEBUG level.

    Returns
    -------
    coords : NDArray[np.floating]
        The coordinates as optimized by SMACOF.
    stress : float
        The stress of the system at the final iteration.
    """
    if verbose:
        log = logger.info
    else:
        log = logger.debug
    i = 0
    npoint = len(distance_matrix)
    stress = stress_func(coords, distance_matrix, weights, ndim)
    for i in range(max_iters):
        if stress == 0:
            break
        _stress = stress

        edm = make_edm(coords)

        B = -1 * weights * distance_matrix / edm
        B[edm == 0] = 0
        B[~np.isfinite(B)] = 0
        B[np.diag_indices_from(B)] -= np.sum(B, axis=1)

        guess = np.dot(B, coords) / npoint

        stress = stress_func(guess, distance_matrix, weights, ndim)
        coords = guess
        if _stress - stress < epsilon:
            break
    log("SMACOF took %d iterations with a final stress of %f", i + 1, stress)
    return coords, stress


def metric_stress(
    coords: NDArray[np.floating],
    distance_matrix: NDArray[np.floating],
    weights: NDArray[np.floating],
    ndim: int,
) -> float:
    """
    Stress that is minimized for metric MDS.

    Parameters
    ----------
    coords : NDArray[np.floating]
        The coordinates to calculate stress at.
        Should be (npoint, ndim)
    distance_matrix : NDArray[np.floating]
        The distance matrix.
        Should be (npoint, npoint), unknown distances should be set to nan.
    weights : NDArray[np.floating]
        How much to weigh each distance in distance_matrix in the metric.
        Should be (npoint, npoint) and have 1-to-1 correspondance with distance_matrix.
    ndim : int, default: 3
        The number of dimensions to scale to.

    Returns
    -------
    stress : float
        The stress of the system at the with the given coordinates.
    """
    npoint = len(distance_matrix)
    idx = np.triu_indices(npoint, 1)
    edm = make_edm(coords.reshape((npoint, ndim)))
    stress = np.sqrt(np.nansum(weights[idx] * (distance_matrix[idx] - edm[idx]) ** 2))
    return stress


def metric_mds(
    distance_matrix: NDArray[np.floating],
    ndim: int = 3,
    weights: Optional[NDArray[np.floating]] = None,
    guess: Optional[NDArray[np.floating]] = None,
    use_smacof: bool = True,
    **kwargs,
) -> NDArray[np.floating]:
    """
    Perform metric MDS.
    This is useful over classical MDS if you have missing values or need to weight distances.

    Parameters
    ----------
    distance_matrix : NDArray[np.floating]
        The distance matrix.
        Should be (npoint, npoint), unknown distances should be set to nan.
    ndim : int, default: 3
        The number of dimensions to scale to.
    weights : Optional[NDArray[np.floating]], default: None
        How much to weigh each distance in distance_matrix in the metric.
        Weights should be finite and non-negative, invalid weights will be set to 0.
        Should be (npoint, npoint) and have 1-to-1 correspondance with distance_matrix.
    guess : Optional[NDArray[np.floating]], default: None
        Initial guess at coordinates.
        Should be (npoint, ndim) and in the same order as distance_matrix.
    use_smacof : bool, default: True
        If True use smacof for the optimization.
        If False use scipy.optimize.minimize.
    **kwargs
        Keyword arguments to pass to smacof or scipy.optimize.minimize.

    Returns
    -------
    coords : NDArray[np.floating]
        The output coordinates, will be (npoint, ndim).
        Points are in the same order as distance_matrix.

    Raises
    ------
    ValueError
        If distance_matrix is not square.
        If the shape of weights or guess is not consistant with distance_matrix.
    """
    npoint = len(distance_matrix)
    if distance_matrix.shape != (npoint, npoint):
        raise ValueError("Distance matrix should be square")

    if weights is None:
        weights = np.ones_like(distance_matrix)
    elif weights.shape != (npoint, npoint):
        raise ValueError("Weights must match distance_matrix")
    else:
        neg_msk = weights < 0
        if np.any(neg_msk):
            logger.warn("Negetive weight found, setting to 0.")
            weights[neg_msk] = 0
        nfin_msk = ~np.isfinite(weights)
        if np.any(nfin_msk):
            logger.warn("Non-finite weight found, setting to 0.")
            weights[nfin_msk] = 0

    if guess is None:
        logger.info("No initial guess provided, using a random set of points.")
        guess = _init_coords(npoint, ndim, distance_matrix)
    elif guess.shape != (npoint, ndim):
        raise ValueError("Guess must be (npoint, ndim)")

    if use_smacof:
        coords, _ = smacof(
            metric_stress, guess, distance_matrix, weights, ndim, **kwargs
        )
    else:
        res = opt.minimize(
            metric_stress,
            guess.ravel(),
            args=(distance_matrix.astype(float), weights.astype(float), ndim),
            **kwargs,
        )
        logger.info(
            "Finished with a final stress of %f\n Optimizer message: %s",
            res.fun,
            res.message,
        )
        coords = res.x.reshape((npoint, ndim))
    return coords


def nonmetric_stress(
    coords: NDArray[np.floating],
    f_dist: NDArray[np.floating],
    weights: NDArray[np.floating],
    ndim: int,
) -> float:
    """
    Stress that is minimized for nonmetric MDS.

    Parameters
    ----------
    coords : NDArray[np.floating]
        The coordinates to calculate stress at.
        Should be (npoint, ndim)
    f_dist : NDArray[np.floating]
        The distance matrix with the isotonic regression applied.
        Should be (npoint, npoint), unknown distances should be set to nan.
    weights : NDArray[np.floating]
        How much to weigh each distance in distance_matrix in the metric.
        Should be (npoint, npoint) and have 1-to-1 correspondance with distance_matrix.
    ndim : int, default: 3
        The number of dimensions to scale to.

    Returns
    -------
    stress : float
        The stress of the system at the with the given coordinates.
    """
    npoint = len(f_dist)
    idx = np.triu_indices(npoint, 1)
    edm = make_edm(coords.reshape((npoint, ndim)))

    num = np.nansum(weights[idx] * (f_dist[idx] - edm[idx]) ** 2)
    denom = np.nansum(edm[idx] ** 2)
    stress = np.sqrt(num / denom)

    return stress


def nonmetric_mds(
    distance_matrix: NDArray[np.floating],
    ndim: int = 3,
    weights: Optional[NDArray[np.floating]] = None,
    guess: Optional[NDArray[np.floating]] = None,
    epsilon_outer: float = 1e-10,
    max_iters_outer: int = 200,
    use_smacof: bool = False,
    **kwargs,
) -> NDArray[np.floating]:
    """
    Perform nonmetric MDS.
    This is useful over metric MDS if you have some wierd dissimilarity
    (ie. not euclidean).

    Parameters
    ----------
    distance_matrix : NDArray[np.floating]
        The distance matrix.
        Should be (npoint, npoint), unknown distances should be set to nan.
    ndim : int, default: 3
        The number of dimensions to scale to.
    weights : Optional[NDArray[np.floating]], default: None
        How much to weigh each distance in distance_matrix in the metric.
        Weights should be finite and non-negative, invalid weights will be set to 0.
        Should be (npoint, npoint) and have 1-to-1 correspondance with distance_matrix.
    guess : Optional[NDArray[np.floating]], default: None
        Initial guess at coordinates.
        Should be (npoint, ndim) and in the same order as distance_matrix.
    epsilon_outer : float, default: 1e-10
        The difference in stress between iterations before stopping outer optimization.
    max_iters_outer : int, default: 200
        Maximum number of iterations for outer optimization.
    use_smacof : bool, default: False
        If True use smacof for the optimization.
        If False use scipy.optimize.minimize.
    **kwargs
        Keyword arguments to pass to smacof or scipy.optimize.minimize.

    Returns
    -------
    coords : NDArray[np.floating]
        The output coordinates, will be (npoint, ndim).
        Points are in the same order as distance_matrix.

    Raises
    ------
    ValueError
        If distance_matrix is not square.
        If the shape of weights or guess is not consistant with distance_matrix.
    """
    if use_smacof:
        logger.warn(
            "You are using SMACOF with nonmetric mds, this doesn't currently work very well..."
        )
        if "verbose" not in kwargs:
            kwargs["verbose"] = False
    npoint = len(distance_matrix)
    if distance_matrix.shape != (npoint, npoint):
        raise ValueError("Distance matrix should be square")

    if weights is None:
        weights = np.ones_like(distance_matrix)
    elif weights.shape != (npoint, npoint):
        raise ValueError("Weights must match distance_matrix")
    else:
        neg_msk = weights < 0
        if np.any(neg_msk):
            logger.warn("Negetive weight found, setting to 0.")
            weights[neg_msk] = 0
        nfin_msk = not np.isfinite(weights)
        if np.any(nfin_msk):
            logger.warn("Non-finite weight found, setting to 0.")
            weights[nfin_msk] = 0

    if guess is None:
        logger.info("No initial guess provided, using a random set of points.")
        guess = _init_coords(npoint, ndim, distance_matrix)
    elif guess.shape != (npoint, ndim):
        raise ValueError("Guess must be (npoint, ndim)")

    idx = np.triu_indices(npoint, 1)
    flat_dist = np.ravel(distance_matrix[idx])
    flat_weight = np.ravel(weights[idx])
    msk = np.isfinite(flat_dist) + np.isfinite(flat_weight)
    flat_dist = flat_dist[msk]
    flat_weight = flat_weight[msk]
    f_dist = np.zeros_like(distance_matrix)

    stress = np.inf
    i = 0
    ir = IsotonicRegression()
    coords = guess.copy()
    for i in range(max_iters_outer):
        _stress = stress
        # Make a guess at f
        edm = make_edm(guess)[idx][msk]
        _f_dist_flat = ir.fit_transform(flat_dist, edm, sample_weight=flat_weight)
        f_dist_flat = np.nan + np.empty(msk.shape)
        f_dist_flat[msk] = _f_dist_flat
        f_dist[idx] = f_dist_flat

        # Solve for the coordinates at this f
        if use_smacof:
            guess, stress = smacof(
                nonmetric_stress, coords, f_dist, weights, ndim, **kwargs
            )
        else:
            res = opt.minimize(
                nonmetric_stress,
                coords.ravel(),
                args=(f_dist.astype(float), weights.astype(float), ndim),
                **kwargs,
            )
            guess = res.x.reshape((npoint, ndim))
            stress = res.fun
        if guess is None:
            raise RuntimeError("Current guess is None, something went wrong...")
        coords = guess
        if _stress - stress < epsilon_outer:
            break
        if stress == 0:
            break
    logger.info("Took %d iterations with a final stress of %f", i + 1, stress)

    return coords
