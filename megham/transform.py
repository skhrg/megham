"""
Functions to computing and working with transformations between point clouds
"""
from typing import Optional

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def get_shift(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    row_basis: bool = True,
    method: str = "median",
    weights: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    Get shift between two point clouds.
    Shift can be applied as dst = src + shift.

    Parameters
    -----------
    src : NDArray[np.floating]
        A (ndim, npoints) array of source points.
    dst : NDArray[np.floating]
        Nominally a (ndim, npoints) array of destination points,
        but really any array broadcastable with src is accepted.
        Some useful options are:
        * np.zeros(1) to align with the origin
        * A (ndim, 1) array to align with an arbitrary point
    row_basis : bool, default: True
        If the basis of the points is row.
        If row basis then each row of src and dst is a point.
        If col basis then each col of src and dst is a point.
    method : str, default: 'median'
        Method to use to align points.
        Current accepted values are: 'median' and 'mean'
    weights : Optional[NDArray[np.floating]], default: None
        (npoints,) array of weights to use.
        If provided and method is 'mean' then a weighted average is used.
        If method is median this is not currently used.

    Returns
    -------
    shift : NDArray[np.floating]
        The (ndim,) shift to apply after transformation.
        If point are in col basis will be returned as a column vector.

    Raises
    ------
    ValueError
        If an invalid method is provided
    """
    if method not in ["median", "mean"]:
        raise ValueError(f"Invalid method: {method}")

    if row_basis:
        src = src.T
        dst = np.atleast_2d(dst).T

    shift = np.zeros(src.shape[0])
    if method == "median":
        shift = np.median(dst - src, axis=-1)
    elif method == "mean":
        if weights is None:
            shift = np.mean(dst - src, axis=-1)
        else:
            wdiff = weights * (dst - src)
            shift = np.nansum(wdiff, axis=1) / np.nansum(weights)

    if not row_basis:
        shift = shift[..., np.newaxis]

    return shift


def get_rigid(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    row_basis: bool = True,
    center_dst: bool = True,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get rigid transformation between two point clouds.
    It is assumed that the point clouds have the same registration,
    ie. src[i] corresponds to dst[i].

    Transformation is dst = src@rot + shift in row basis,
    and dst = rot@src + shift in col basis.

    Parameters
    ----------
    src : NDArray[np.floating]
        A (ndim, npoints) array of source points.
    dst : NDArray[np.floating]
        A (ndim, npoints) array of destination points.
    row_basis : bool, default: True
        If the basis of the points is row.
        If row basis then each row of src and dst is a point.
        If col basis then each col of src and dst is a point.
    center_dst : bool, default: True
        If True, dst will be recentered at the origin before computing transformation.
        This is done with get_shift, but weights will not be used if provided.
    **kwargs
        Arguments to pass to get_shift.

    Returns
    -------
    rotation : NDArray[np.floating]
        The (ndim, ndim) rotation matrix.
    shift : NDArray[np.floating]
        The (ndim,) shift to apply after transformation.
        If point are in col basis will be returned as a column vector.

    Raises
    ------
    ValueError
        If the input point clouds have different shapes.
        If the input point clouds don't have enough points.
    """
    if src.shape != dst.shape:
        raise ValueError("Input point clouds should have the same shape")
    if row_basis:
        src = src.T
        dst = dst.T

    msk = np.isfinite(src).all(axis=0) * np.isfinite(dst).all(axis=0)
    ndim = len(src)
    if np.sum(msk) < ndim * (ndim - 1) / 2:
        raise ValueError("Not enough finite points to compute transformation")

    _dst = dst[:, msk].copy()
    if center_dst:
        _kwargs = kwargs.copy()
        _kwargs.update({"weights": None})
        _dst += get_shift(_dst, np.zeros(1), False, **_kwargs)
    _src = src[:, msk].copy()
    _src += get_shift(_src, _dst, False, **kwargs)

    M = _src @ (_dst.T)
    u, _, vh = la.svd(M)
    v = vh.T
    uT = u.T

    corr = np.eye(ndim)
    corr[-1, -1] = la.det((v) @ (uT))
    rot = v @ corr @ uT

    transformed = rot @ src[:, msk]
    shift = get_shift(transformed, dst[:, msk], False, **kwargs)

    if row_basis:
        rot = rot.T
        shift = shift[:, 0]

    return rot, shift


def get_affine(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    row_basis: bool = True,
    center_dst: bool = True,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get affine transformation between two point clouds.
    It is assumed that the point clouds have the same registration,
    ie. src[i] corresponds to dst[i].

    Transformation is dst = src@affine + shift in row basis,
    and dst = affine@src + shift in col basis.

    Parameters
    ----------
    src : NDArray[np.floating]
        A (npoints, ndim) or (ndim, npoints) array of source points.
    dst : NDArray[np.floating]
        A ((npoints, ndim) or (ndim, npoints) array of destination points.
    row_basis : bool, default: True
        If the basis of the points is row.
        If row basis then each row of src and dst is a point.
        If col basis then each col of src and dst is a point.
    center_dst : bool, default: True
        If True, dst will be recentered at the origin before computing transformation.
        This is done with get_shift, but weights will not be used if provided.
    **kwargs
        Arguments to pass to get_shift.

    Returns
    -------
    affine : NDArray[np.floating]
        The (ndim, ndim) transformation matrix.
    shift : NDArray[np.floating]
        The (ndim,) shift to apply after transformation.
        If point are in col basis will be returned as a column vector.

    Raises
    ------
    ValueError
        If the input point clouds have different shapes.
        If the input point clouds don't have enough points.
    """
    if src.shape != dst.shape:
        raise ValueError("Input point clouds should have the same shape")
    if row_basis:
        src = src.T
        dst = dst.T

    msk = np.isfinite(src).all(axis=0) * np.isfinite(dst).all(axis=0)
    if np.sum(msk) < len(src) + 1:
        raise ValueError("Not enough finite points to compute transformation")

    _dst = dst[:, msk].copy()
    if center_dst:
        _kwargs = kwargs.copy()
        _kwargs.update({"weights": None})
        _dst += get_shift(_dst, np.zeros(1), False, **_kwargs)
    _src = src[:, msk].copy()
    _src += get_shift(_src, _dst, False, **kwargs)

    M = np.vstack((_src, _dst)).T
    *_, vh = la.svd(M)
    vh_splits = [
        quad for half in np.split(vh.T, 2, axis=0) for quad in np.split(half, 2, axis=1)
    ]
    affine = np.dot(vh_splits[2], la.pinv(vh_splits[0]))

    transformed = affine @ src[:, msk]
    shift = get_shift(transformed, dst[:, msk], False, **kwargs)

    if row_basis:
        affine = affine.T
        shift = shift[:, 0]

    return affine, shift


def apply_transform(
    src: NDArray[np.floating],
    transform: NDArray[np.floating],
    shift: NDArray[np.floating],
    row_basis: bool = True,
) -> NDArray[np.floating]:
    """
    Apply a transformation to a set of points.

    Parameters
    ----------
    src : NDArray[np.floating]
        The points to transform.
        Should have shape (ndim, npoints) or (npoints, ndim).
    transform: NDArray[np.floating]
        The transformation matrix.
        Should have shape (ndim, ndim).
    shift : NDArray[np.floating]
        The shift to apply after the affine tranrform.
        Should have shape (ndim,).
    row_basis : bool, default: True
        Whether or not the input and output need to be transposed.
        This is the case when src is (npoints, ndim).
        By default the function will try to figure this out in its own,
        this is only used in the case where it can't because src is (ndim, ndim).

    Returns
    -------
    transformed : NDArray[np.floating]
        The transformed points.
        Has the same shape as src.

    Raises
    ------
    ValueError
        If src is not a 2d array.
        If one of src's axis is not of size ndim.
        If affine and shift have inconsistent shapes.
    """
    ndim = len(shift)
    if transform.shape != (ndim, ndim):
        raise ValueError(
            f"From shift we assume ndim={ndim} but transform has shape {transform.shape}"
        )
    src_shape = np.array(src.shape)
    if len(src_shape) != 2:
        raise ValueError(f"src should be a 2d array, not {len(src.shape)}d")

    if row_basis:
        transformed = src @ transform + shift
    else:
        transformed = transform @ src + shift
    return transformed


def decompose_affine(
    affine: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Decompose an affine transformation into its components.
    This decomposetion treats the affine matrix as: rotation * shear * scale.

    Parameters
    ----------
    affine : NDArray[np.floating]
        The (ndim, ndim) affine transformation matrix.

    Returns
    -------
    scale : NDArray[np.floating]
        The (ndim,) array of scale parameters.
    shear : NDArray[np.floating]
        The (ndim*(ndim - 1)/2,) array of shear parameters.
    rot: NDArray[np.floating]
        The (ndim, ndim) rotation matrix.
        If ndim is 2 or 3 then decompose_rotation can be used to get euler angles.

    Raises
    ------
    ValueError
        If affine is not ndim by ndim.
    """
    ndim = len(affine)
    if affine.shape != (ndim, ndim):
        raise ValueError("Affine matrix should be ndim by ndim")
    # Use the fact that rotation matrix times its transpose is the identity
    no_rot = affine.T @ affine
    # Decompose to get a matrix with just scale and shear
    no_rot = la.cholesky(no_rot).T

    scale = np.diag(no_rot)
    shear = (no_rot / scale[:, None])[np.triu_indices(len(no_rot), k=1)]
    rot = affine @ la.inv(no_rot)

    return scale, shear, rot


def decompose_rotation(rotation: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Decompose a rotation matrix into its xyz rotation angles.
    This currently won't work on anything higher than 3 dimensions.

    Parameters
    ----------
    rotation : NDArray[np.floating]
        The (ndim, ndim) rotation matrix.

    Returns
    -------
    angles : NDArray[np.floating]
        The rotation angles in radians.
        If the input is 3d then this has 3 angles in xyz order,
        if 2d it just has one.

    Raises
    ------
    ValueError
        If affine is not ndim by ndim.
        If ndim is not 2 or 3.
    """
    ndim = len(rotation)
    if ndim > 3:
        raise ValueError("No support for rotations in more than 3 dimensions")
    if ndim < 2:
        raise ValueError("Rotations with less than 2 dimensions don't make sense")
    if rotation.shape != (ndim, ndim):
        raise ValueError("Rotation matrix should be ndim by ndim")
    _rotation = np.eye(3)
    _rotation[:ndim, :ndim] = rotation
    angles = R.from_matrix(_rotation).as_euler("xyz")

    if ndim == 2:
        angles = angles[-1:]
    return angles
