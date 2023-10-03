"""
Functions for doing multidimensional scaling.
"""
import warnings
from typing import Optional

import numpy as np
import scipy.optimize as opt
import scipy.spatial.distance as dist
from numpy.typing import NDArray


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
    **kwargs
        Keyword arguments to pass so scipy.optimize.minimize.

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
            warnings.warn("Negetive weight found, setting to 0.")
            weights[neg_msk] = 0
        nfin_msk = not np.isfinite(weights)
        if np.any(nfin_msk):
            warnings.warn("Non-finite weight found, setting to 0.")
            weights[nfin_msk] = 0

    if guess is None:
        warnings.warn(
            "No initial guess provided, it is unlikey that you will get a good result."
        )
        guess = np.zeros((npoint, ndim))
    elif guess.shape != (npoint, ndim):
        raise ValueError("Guess must be (npoint, ndim)")

    res = opt.minimize(
        metric_stress,
        guess.ravel(),
        args=(distance_matrix.astype(float), weights.astype(float), ndim),
        **kwargs,
    )
    print(f"Optimizer success: {res.success}\n Optimizer message: {res.message}")
    return res.x.reshape((npoint, ndim))
