"""
Functions for doing multidimensional scaling.
"""
import numpy as np
import scipy.spatial.distance as dist
from numpy.typing import NDArray


def make_edm(coords: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Make an Euclidean distance matrix from a set of points.

    Parameters
    ----------
    coords : NDArray[np.floating]
        The (npoints, ndim) array of input points.

    Returns
    -------
    edm : NDArray[np.floating]
        The (npoints, npoints) euclidean distance matrix.
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
    ndim: int, default: 3
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
