from typing import Optional, Sequence

import numpy as np
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


def estimate_var(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    dim_groups: Optional[Sequence[Sequence[int] | NDArray[np.int_]]] = None,
) -> NDArray[np.floating]:
    """
    Estimate variance between point clouds for use with something like a GMM.

    Parameters
    ----------
    src : NDArray[np.floating]
        The set of source points to be mapped onto the target points.
        Should have shape (nsrcpoints, ndim).
    dst : NDArray[np.floating]
        The set of destination points to be mapped onto.
        Should have shape (ndstpoints, ndim).
    dim_groups : Optional[Sequence[Sequence[int] | NDArray[np.int_]]], default: None
        Which dimensions should be computed together.
        If None all dimensions will be treated seperately.

    Returns
    -------
    var : NDArray[np.floating]
        The estimated variance.
        Will have shape (ndim,).
    """
    nsrcpoints, ndim = src.shape
    ndstpoints = len(dst)

    if dim_groups is None:
        dim_groups = [[dim] for dim in range(ndim)]
    else:
        dims_flat = np.concatenate(dim_groups)
        no_group = np.setdiff1d(np.arange(ndim), dims_flat)
        dim_groups = list(dim_groups)
        dim_groups = dim_groups + [[dim] for dim in no_group]

    var = np.zeros(ndim)
    for dim_group in dim_groups:
        sq_diff = dist.cdist(src[:, dim_group], dst[:, dim_group], metric="sqeuclidean")
        var[dim_group] = np.nansum(sq_diff) / (len(dim_group) * nsrcpoints * ndstpoints)

    return var


def estimate_spacing(coords: NDArray[np.floating]) -> float:
    """
    Estimate the spacing between points in a point cloud.
    This is just the median distance between nearest neighbors.

    Parameters
    ----------
    coords : NDArray[np.floating]
        The point cloud to estimate spacing of.
        Should have shape (npoint, ndim).

    Returns
    -------
    spacing : float
        The spacing between points.
    """
    edm = make_edm(coords)
    edm[edm == 0] = np.nan
    nearest_dists = np.nanmin(edm, axis=0)

    return np.median(nearest_dists)


def gen_weights(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    var: Optional[NDArray[np.floating]] = None,
    pdf: bool = False,
) -> NDArray[np.floating]:
    """
    Generate weights between points in two registered point clouds.
    The weight here is just the liklihood from a gaussian.
    Note that this is not a GMM, each weight is computed from a single
    gaussian since we are assuming a known registration.

    Parameters
    ----------
    src : NDArray[np.floating]
        The set of source points to be mapped onto the target points.
        Should have shape (nsrcpoints, ndim).
    dst : NDArray[np.floating]
        The set of destination points to be mapped onto.
        Should have shape (ndstpoints, ndim).
    var : Optional[NDArray[np.floating]], default: None
        The variance along each axis.
        Should have shape (ndim,) if provided.
        If None, will be computed with estimate_var
    pdf : bool, default: False
        If True apply the 1/sqrt(2*pi*var) normalization factor.
        This makes the weights the PDF of a normal distribution.

    Returns
    -------
    weights : NDArray[np.floating]
        (npoints,) array of weights.
    """
    if var is None:
        var = estimate_var(src, dst)
    norm = np.ones_like(var)
    if pdf:
        norm = 1.0 / np.sqrt(2 * np.pi * var)

    # Compute nd gaussian for each pair
    weights = np.prod(norm * np.exp(-0.5 * (src - dst) ** 2 / var), axis=1)

    return weights
