"""
Functions to computing and working with transformations between point clouds
"""
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


def get_affine(
    src: NDArray[np.floating], dst: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get affine transformation between two point clouds.
    It is assumed that the point clouds have the same registration,
    ie. src[i] corresponds to dst[i].

    Transformation is dst = affine@src + shift

    Parameters
    ----------
    src : NDArray[np.floating]
         A (ndim, npoints) array of source points.
    dst : NDArray[np.floating]
         A (ndim, npoints) array of destination points.

    Returns
    -------
    affine : NDArray[np.floating]
        The (ndim, ndim) transformation matrix.
    shift : NDArray[np.floating]
        The (ndim,) shift to apply after transformation.

    Raises
    ------
    ValueError
        If the input point clouds have different shapes.
        If the input point clouds don't have enough points.
    """
    if src.shape != dst.shape:
        raise ValueError("Input point clouds should have the same shape")
    msk = np.isfinite(src).all(axis=0) * np.isfinite(dst).all(axis=0)
    if np.sum(msk) < len(src) + 1:
        raise ValueError("Not enough finite points to compute transformation")

    M = np.vstack(
        (
            src[:, msk] - np.median(src[:, msk], axis=1)[:, None],
            dst[:, msk] - np.median(dst[:, msk], axis=1)[:, None],
        )
    ).T
    *_, vh = la.svd(M)
    vh_splits = [
        quad for half in np.split(vh.T, 2, axis=0) for quad in np.split(half, 2, axis=1)
    ]
    affine = np.dot(vh_splits[2], la.pinv(vh_splits[0]))

    transformed = affine @ src[:, msk]
    shift = np.median(dst[:, msk] - transformed, axis=1)

    return affine, shift


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
