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
        * A (ndim,) array to align with an arbitrary point
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

    Raises
    ------
    ValueError
        If an invalid method is provided
    """
    if method not in ["median", "mean"]:
        raise ValueError(f"Invalid method: {method}")

    shift = np.zeros(src.shape[1])
    if method == "median":
        shift = np.median(dst - src, axis=0)
    elif method == "mean":
        if weights is None:
            shift = np.mean(dst - src, axis=0)
        else:
            wdiff = weights[..., None] * (dst - src)
            shift = np.nansum(wdiff, axis=0) / np.nansum(weights)

    return shift


def get_rigid(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    center_dst: bool = True,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get rigid transformation between two point clouds.
    It is assumed that the point clouds have the same registration,
    ie. src[i] corresponds to dst[i].

    Transformation is dst = src@rot + shift.

    Parameters
    ----------
    src : NDArray[np.floating]
        A (npoints, ndim) array of source points.
    dst : NDArray[np.floating]
        A (npoints, ndim) array of destination points.
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

    msk = np.isfinite(src).all(axis=1) * np.isfinite(dst).all(axis=1)
    ndim = src.shape[1]
    if np.sum(msk) < ndim * (ndim - 1) / 2:
        raise ValueError("Not enough finite points to compute transformation")

    _dst = dst[msk].copy()
    if center_dst:
        _kwargs = kwargs.copy()
        _kwargs.update({"weights": None})
        _dst += get_shift(_dst, np.zeros(1), **_kwargs)
    _src = src[msk].copy()
    _src += get_shift(_src, _dst, **kwargs)

    M = _src.T @ (_dst)
    u, _, vh = la.svd(M)
    v = vh.T
    uT = u.T

    corr = np.eye(ndim)
    corr[-1, -1] = la.det((v) @ (uT))
    rot = v @ corr @ uT
    rot = rot.T

    transformed = src[msk] @ rot
    shift = get_shift(transformed, dst[msk], **kwargs)

    return rot, shift


def get_affine(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    weights: Optional[NDArray[np.floating]] = None,
    center_dst: bool = True,
    force_svd: bool = False,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get affine transformation between two point clouds.
    It is assumed that the point clouds have the same registration,
    ie. src[i] corresponds to dst[i].

    Transformation is dst = src@affine + shift.

    Parameters
    ----------
    src : NDArray[np.floating]
        A (npoints, ndim) array of source points.
    dst : NDArray[np.floating]
        A (npoints, ndim) array of destination points.
    weights : Optional[NDArray[np.floating]], default: None
        (npoints,) array of weights to use.
        If provided a weighted least squares is done instead of an SVD.
    center_dst : bool, default: True
        If True, dst will be recentered at the origin before computing transformation.
        This is done with get_shift, but weights will not be used if provided.
    force_svd : bool, default: False
        If True the SVD is used even if there are a small number of points
        or weights are present.
    **kwargs
        Arguments to pass to get_shift.

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

    msk = np.isfinite(src).all(axis=1) * np.isfinite(dst).all(axis=1)
    if np.sum(msk) < src.shape[1] + 1:
        raise ValueError("Not enough finite points to compute transformation")

    # When we have a small number of points lstsq is better than SVD
    # Condition is a bit arbitrary for now
    if force_svd is False and weights is None and np.sum(msk) < 50 * src.shape[1]:
        weights = np.ones(len(src))

    _dst = dst[msk].copy()
    if center_dst:
        _dst += get_shift(_dst, np.zeros(1), **kwargs)
    _src = src[msk].copy()
    init_shift = get_shift(_src, _dst, weights=weights, **kwargs)

    if force_svd or weights is None:
        M = np.vstack((_src.T, (_dst - init_shift).T)).T
        *_, vh = la.svd(M)
        vh_splits = [
            quad
            for half in np.split(vh.T, 2, axis=0)
            for quad in np.split(half, 2, axis=1)
        ]
        affine = np.dot(vh_splits[2], la.pinv(vh_splits[0])).T
        shift = init_shift
    else:
        rt_weight = np.sqrt(weights[msk])[..., None]
        wsrc = rt_weight * _src
        wdst = rt_weight * (_dst - init_shift)
        x, *_ = la.lstsq(
            np.column_stack((wsrc, np.ones(len(wsrc)))), wdst, check_finite=False
        )
        affine = x[:-1]
        shift = x[-1] + init_shift

    transformed = src[msk] @ affine + shift
    shift += get_shift(transformed, dst[msk], **kwargs)

    return affine, shift


def get_affine_two_stage(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    weights: Optional[NDArray[np.floating]],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get affine transformation between two point clouds with a two stage solver.
    This first uses the SVD to do an intitial alignment and
    then uses weighted least squares to compute a correction on top of that.

    Transformation is dst = affine@src + shift

    Parameters
    ----------
    src : NDArray[np.floating]
        A (npoints, ndim) array of source points.
    dst : NDArray[np.floating]
        A (npoints, ndim) array of destination points.
    weights : NDArray[np.floating]
        (npoints,) array of weights to use.
        If provided a weighted least squares is done instead of an SVD.

    Returns
    -------
    affine : NDArray[np.floating]
        The (ndim, ndim) transformation matrix.
    shift : NDArray[np.floating]
        The (ndim,) shift to apply after transformation.
    """
    if weights is None:
        weights = np.ones(len(src))
    # Do an initial rigid alignment
    affine_0, shift_0 = get_rigid(src, dst, method="mean")  # force_svd=True)
    init_align = apply_transform(src, affine_0, shift_0)
    # Do an alignment without weights
    affine_1, shift_1 = get_affine(init_align, dst, force_svd=True, method="mean")
    init_align = apply_transform(init_align, affine_1, shift_1)
    # Now compute the actual transform
    affine, shift = get_affine(init_align, dst, weights)
    # Compose the transforms
    affine, shift = compose_transform(
        *compose_transform(affine_0, shift_0, affine_1, shift_1), affine, shift
    )
    # Now one last shift correction
    transformed = apply_transform(src, affine, shift)
    shift += get_shift(transformed, dst, "mean", weights)

    return affine, shift


def apply_transform(
    src: NDArray[np.floating],
    transform: NDArray[np.floating],
    shift: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Apply a transformation to a set of points.

    Parameters
    ----------
    src : NDArray[np.floating]
        The points to transform.
        Should have shape  (npoints, ndim).
    transform: NDArray[np.floating]
        The transformation matrix.
        Should have shape (ndim, ndim).
    shift : NDArray[np.floating]
        The shift to apply after the affine tranrform.
        Should have shape (ndim,).

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

    transformed = src @ transform + shift
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


def compose_transform(
    transform_1: NDArray[np.floating],
    shift_1: NDArray[np.floating],
    transform_2: NDArray[np.floating],
    shift_2: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Combine transformations to get one that is equivalent to:
    dst = (src@transform_1 + shift)@transform_2 + shift_2

    Parameters
    ----------
    transform_1 : NDArray[np.floating]
        The first transform (affine or rotation matrix).
        Should have shape (ndim, ndim).
    shift_1 : NDArray[np.floating]
        The first shift.
        Should have shape (ndim,).
    transform_2 : NDArray[np.floating]
        The second transform (affine or rotation matrix).
        Should have shape (ndim, ndim).
    shift_2 : NDArray[np.floating]
        The second shift.
        Should have shape (ndim,).

    Returns
    -------
    transform : NDArray[np.floating]
        The composed transform.
        Has shape (ndim, ndim).
    shift : NDArray[np.floating].
        The composed shift.
        Has shape (ndim,).
    """
    transform = transform_1 @ transform_2
    shift = shift_1 @ transform_2 + shift_2

    return transform, shift


def decompose_transform(
    transform: NDArray[np.floating],
    shift: NDArray[np.floating],
    transform_1: NDArray[np.floating],
    shift_1: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Decompose transformations to get one with the other removed.
    This is solving for transform_2 and shift_2 in the following equation:
    dst = src@transform + shift = (src@transform_1 + shift)@transform_2 + shift_2

    Parameters
    ----------
    transform : NDArray[np.floating]
        The composed transform (affine or rotation matrix).
        Should have shape (ndim, ndim).
    shift : NDArray[np.floating]
        The composed shift.
        Should have shape (ndim,)
    transform_1 : NDArray[np.floating]
        The transform (affine or rotation matrix) to remove.
        Should have shape (ndim, ndim).
    shift_1 : NDArray[np.floating]
        The shift to remove.
        Should have shape (ndim,)

    Returns
    -------
    transform_2 : NDArray[np.floating]
        The transform with the first transform removed.
        Has shape (ndim, ndim).
    shift_2 : NDArray[np.floating].
        The shift with the first transform removed.
        Has shape (ndim,).
    """
    transform_2 = np.linalg.inv(transform_1) @ transform
    shift_2 = shift - shift_1 @ transform_2

    return transform_2, shift_2


def invert_transform(
    transform: NDArray[np.floating], shift: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Invert a transformation.
    If the inverted transformation is applied to a point cloud that has already been
    transformed, you will recover the original point cloud.

    Parameters
    ----------
    transform : NDArray[np.floating]
        The transform (affine or rotation matrix) to invert.
        Should have shape (ndim, ndim).
    shift : NDArray[np.floating]
        The shift to invert.
        Should have shape (ndim,)

    Returns
    -------
    transform_inv : NDArray[np.floating]
        The inverted transformation matrix.
    shift_inv : NDArray[np.floating]
        The inverted shift vector.
    """
    transform_inv = np.linalg.inv(transform)
    shift_inv = (-1 * shift) @ transform_inv

    return transform_inv, shift_inv


def approx_common_mode_svd(
    transforms: list[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """
    Approximate the common mode rotation between matrices by using the SVD.
    This is done by taking a naive mean of all matrices and then finding the nearest
    orthogonal matrix to this mean matrix: $U \tilde{I} V^T$ where $U$ and $V^T$ are
    calculated from the SVD as usual and $\tilde{I}$ is the identity with its last element
    (ie: the bottom right) replaced by $det(U)det(V)$.


    Parameters
    ----------
    transforms : list[NDArray[np.floating]]
        The transforms (affine or rotation matrix) to calculate the commom mode for.
        Each should have shape (ndim, ndim).

    Returns
    -------
    common_rot : NDArray[np.floating]
        The approximate common mode rotation.
    """
    mean_mat = np.mean(transforms, axis=0)
    u, _, vh = la.svd(mean_mat.T)
    s = np.eye(len(mean_mat))
    s[-1, -1] = la.det(u) * la.det(vh.T)

    common_rot = u @ s @ vh
    return common_rot.T


def get_common_mode(
    transforms: list[NDArray[np.floating]],
    shifts: list[NDArray[np.floating]],
    reference: int = -2,
    rigid_only: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Get the rigid common mode from a set of transformations and shift.
    This is done by computing the matrix exponential with respect to a reference matrix
    and then computing the arithithmatic mean in corresponding Lie group before transforming back.

    The choice of the reference matrix can effect the quality of this calculation,
    since the Lie group will live in the tangent space of the reference matrix.
    The farther the reference is from the input matrices the more unstable this calculation is.
    If your martrices are very different (ie. not much of a common mode) it is possible to get
    odd results from this.
    By default `approx_common_mode_svd` will be used to generate the reference matrice,
    this will genrally work well unless your transformations have basically no common mode.

    Parameters
    ----------
    transforms : list[NDArray[np.floating]]
        The transforms (affine or rotation matrix) to calculate the commom mode for.
        Each should have shape (ndim, ndim).
    shift : NDArray[np.floating]
        The shifts to calculate the commom mode for.
        Each should have shape (ndim,)
    reference : int, default: -2
        The matrix whose tangent space we calculate the common mode in.
        If this is -2 then `approx_common_mode_svd` is called to generate the reference.
        If this is -1 then the indentity is used.
        If this is 0 or greater then `transforms[reference]` is used.
    rigid_only : bool, default: True
        If True then return only the rigid portion of the transform.

    Returns
    -------
    common_transform : NDArray[np.floating]
        The common mode portion of the input transforms.
    common_shift : NDArray[np.floating]
        The common mode portion of the input shifts.
    """
    if reference >= 0:
        ref_transform = transforms[reference]
    elif reference == -1:
        ref_transform = np.eye(len(transforms[0]))
    elif reference == -2:
        ref_transform = approx_common_mode_svd(transforms)
    else:
        raise ValueError("Invalid reference")

    # Compute the average in the Lie group
    avg_group = np.mean(np.log(ref_transform.T @ np.array(transforms)), axis=0)

    # Now back to the Lie algebra
    common_transform = ref_transform @ np.exp(avg_group)

    # Keep only the rotation if we want
    if rigid_only:
        _, _, common_transform = decompose_affine(common_transform)

    # For the shift we can do the easy thing
    common_shift = np.mean(shifts, axis=0)

    return common_transform, common_shift
