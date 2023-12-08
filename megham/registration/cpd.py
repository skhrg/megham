"""
Module for performing coherent point drift
"""
from typing import Optional, Protocol, Sequence

import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist
from numpy.typing import NDArray

from ..transform import apply_transform

epsilon = 1e-7


class Callback(Protocol):
    """
    Callback idea shamelessly stolen from pycpd.
    """

    def __call__(
        self,
        target: NDArray[np.floating],
        transformed: NDArray[np.floating],
        iteration: int,
        err: float,
    ):
        ...


def dummy_callback(target, transformed, iteration, err):
    return


def _init_var(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    dim_groups: Sequence[Sequence[int] | NDArray[np.int_]],
) -> NDArray[np.floating]:
    """
    Initialize the variance used by joint CPD.

    Parameters
    ----------
    source : NDArray[np.floating]
        The set of source points to be mapped onto the target points.
        Should have shape (nsrcpoints, ndim).
    target : NDArray[np.floating]
        The set of target points to be mapped onto.
        Should have shape (ntrgpoints, ndim).
    dim_groups : Sequence[Sequence[int] | NDArray[np.int_]]
        Which dimensions should be transformed together.

    Returns
    -------
    var : NDArray[np.floating]
        The initial variance.
        Will have shape (ndim,).
    """
    nsrcpoints, ndim = source.shape
    ntrgpoints = len(target)

    dims_flat = np.concatenate(dim_groups)
    no_group = np.setdiff1d(np.arange(ndim), dims_flat)
    dim_groups = list(dim_groups)
    dim_groups = dim_groups + [[dim] for dim in no_group]

    var = np.zeros(ndim)
    for dim_group in dim_groups:
        sq_diff = dist.cdist(
            source[:, dim_group], target[:, dim_group], metric="sqeuclidean"
        )
        var[dim_group] = np.sum(sq_diff) / (len(dim_group) * nsrcpoints * ntrgpoints)

    return var


def compute_P(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    var: NDArray[np.floating],
    w: float,
) -> NDArray[np.floating]:
    """
    Compute matrix of probabilities of matches between points in source and target.

    Parameters
    ----------
    source : NDArray[np.floating]
        The set of source points to be mapped onto the target points.
        Nominally when you run this it will be the source points transformed to line up with target.
        Should have shape (nsrcpoints, ndim).
    target : NDArray[np.floating]
        The set of target points to be mapped onto.
        Should have shape (ntrgpoints, ndim).
    var : NDArray[np.floating]
        The variance of the gaussian mixture model.
        Should have shape (ndim,).
    w : float, default: 0.0
        The weight of the uniform distrubution.

    Returns
    -------
    P : NDArray[np.floating]
        Probability matrix of matches between the source and target points.
        P[i,j] is the probability that source[i] corresponds to target[j].
        Has shape (nsrcpoints, ntrgpoints).
    """
    # TODO: implement fast gaussian transform
    nsrcpoints, ndim = source.shape
    ntrgpoints = len(target)

    uni = (
        (w / (1 - w))
        * (nsrcpoints / ntrgpoints)
        * np.sqrt(((2 * np.pi) ** ndim) * np.product(var))
    )
    gaussians = np.ones((nsrcpoints, ntrgpoints))
    for dim in range(ndim):
        sq_diff = dist.cdist(
            np.atleast_2d(source[:, dim]).T,
            np.atleast_2d(target[:, dim]).T,
            metric="sqeuclidean",
        )
        gaussians *= np.exp(-0.5 * sq_diff / var[dim])
    norm_fac = np.clip(np.sum(gaussians, axis=0), epsilon, None)

    P = gaussians / (norm_fac + uni)

    return P


def solve_transform(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    P: NDArray[np.floating],
    dim_groups: Sequence[NDArray[np.int_]],
    cur_var: NDArray[np.floating],
    method: str = "affine",
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], float]:
    """
    Solve for the transformation at each iteration of CPD.

    Parameters
    ----------
    source : NDArray[np.floating]
        The set of source points to be mapped onto the target points.
        Should have shape (nsrcpoints, ndim).
    target : NDArray[np.floating]
        The set of target points to be mapped onto.
        Should have shape (ntrgpoints, ndim).
    P : NDArray[np.floating]
        Probability matrix of matches between the source and target points.
    dim_groups : Sequence[NDArray[np.int_]]
        Which dimensions should be transformed together.
    cur_var : float
        The current mixture model variance.
    method : str, default: 'affine'
        The type of transformation to compute.
        Acceptable values are: affine, rigid.
        If any other value is passed then transform will be the identity.

    Returns
    -------
    transform : NDArray[np.floating]
        Transformation between source and target.
    shift : NDArray[np.floating]
        Shift to be applied after the affine transformation.
    var : NDArray[np.floating]
        Current variance of the mixture model.
    err : float
        Current value of the error function.
    """
    ndim = source.shape[1]
    N_P = np.sum(P)
    PT1 = np.diag(np.sum(P, axis=0))
    P1 = np.diag(np.sum(P, axis=1))

    mu_src = np.sum(P.T @ source, axis=0) / N_P
    mu_trg = np.sum(P @ target, axis=0) / N_P

    src_hat = source - mu_src
    trg_hat = target - mu_trg

    transform = np.eye(ndim)
    shift = np.zeros(ndim)
    new_var = cur_var.copy()
    err = 0
    for dim_group in dim_groups:
        mu_s = mu_src[dim_group]
        mu_t = mu_trg[dim_group]
        src = src_hat[:, dim_group]
        trg = trg_hat[:, dim_group]

        all_mul = trg.T @ P.T @ src
        src_mul = src.T @ P1 @ src

        if method == "affine":
            tfm = np.linalg.solve(src_mul.T, all_mul.T)
        elif method == "rigid":
            U, _, V = la.svd(all_mul, full_matrices=True)
            corr = np.eye(len(dim_group))
            corr[-1, -1] = la.det((V) @ (U))
            tfm = U @ corr @ V
        else:
            tfm = np.eye(ndim)

        sft = mu_t.T - tfm.T @ mu_s.T

        transform[dim_group[:, np.newaxis], dim_group] = tfm
        shift[dim_group] = sft

        trc_trf_mul = np.trace(tfm @ src_mul @ tfm)
        trc_trg_mul = np.trace(trg.T @ PT1 @ trg)
        trc_all = np.trace(all_mul @ tfm)

        var = (trc_trg_mul - trc_all) / (N_P * len(dim_group))
        if var <= 0:
            var = epsilon
        new_var[dim_group] = var

        err += (trc_trg_mul - 2 * trc_all + trc_trf_mul) / (
            2 * cur_var[dim_group][0]
        ) + 0.5 * N_P * len(dim_group) * np.log(cur_var[dim_group][0])

    return transform, shift, new_var, err


def joint_cpd(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    dim_groups: Optional[Sequence[Sequence[int] | NDArray[np.int_]]] = None,
    w: float = 0.0,
    eps: float = 1e-10,
    max_iters: int = 500,
    callback: Callback = dummy_callback,
    method: str = "affine",
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Compute the joint CPD.

    This is an extension of the base CPD algorithm that allows you to fit for
    transforms that treat groups of dimensions seperately but use a jointly computed
    probability matrix and variance.
    This is useful in cases where you have information about a set of points that is correlated
    but in a different basis so you want to avoid degrees of freedom that mix the disprate baseis.
    For example if you have a set of points where you have both spatial information as well as some
    non-spatial scalar associated with each point (ie: frequency) you can use this to compute a
    registration that considers these jointly but transforms them seperately to avoid a scenario
    where you have a rotation between a spatial and a non-spatial dimension.

    Parameters
    ----------
    source : NDArray[np.floating]
        The set of source points to be mapped onto the target points.
        Should have shape (nsrcpoints, ndim).
    target : NDArray[np.floating]
        The set of target points to be mapped onto.
        Should have shape (ntrgpoints, ndim).
    dim_groups : Optional[Sequence[Sequence[int] | NDArray[np.int_]]], default: None
        Which dimensions should be transformed together.
        Each element in this sequence should be a sequence of array of ints that correspond to the
        columns of source and target that are transformed together.
        Any columns that are not included here will not be transformed but will still be used when
        computing the probability and variance.
        None of the elements in this sequence can have overlap.
        If set to None all the dimensions will be transformed together.
    w : float, default: 0.0
        The weight of the uniform distrubution.
        Set higher to reduce sensitivity to noise and outliers at the expense
        of potentially worse performance.
        Should be in the range [0, 1), if not it will snap to one of the bounds.
    eps : float, default: 1e-10
        The convergence criteria.
        When the change in the objective function is less than or equal to this we stop.
    max_iters : int, default: 500
        The maximum number of iterations to run for.
    callback: Callback, default: dummy_callback
        Function that runs once per iteration, can be used to visualize the match process.
        See the Callback Protocol for details on the expected signature.
    method : str, default: 'affine'
        The type of transformation to compute.
        Acceptable values are: affine, rigid.
        If any other value is passed then transform will be the identity.

    Returns
    -------
    transform : NDArray[np.floating]
        The transform transformation that takes source to target.
        Apply using megham.transform.apply_transform.
        Has shape (ndim, ndim).
    shift : NDArray[np.floating]
        The transformation that takes source to target after transform is applied.
        Apply using megham.transform.apply_transform.
        Has shape (ndim,).
    transformed : NDArray[np.floating]
        Source transformed to align with target.
        Has shape (nsrcpoints, ndim).
    P : NDArray[np.floating]
        Probability matrix of matches between the source and target points.
        P[i,j] is the probability that source[i] corresponds to target[j].
        Has shape (nsrcpoints, ntrgpoints).

    Raises
    ------
    ValueError
        If source and target don't share ndim.
        If dim_groups has repeated dimensions or invalid dimensions.
    """
    ndim = source.shape[1]
    if target.shape[1] != ndim:
        raise ValueError(
            f"Source and target don't have same ndim ({ndim} vs {target.shape[1]})"
        )
    if dim_groups is None:
        dim_groups = [
            np.arange(ndim),
        ]
    else:
        dims_flat = np.concatenate(dim_groups)
        dims_bad = dims_flat[(dims_flat < 0) + (dims_flat >= ndim)]
        if len(dims_bad):
            raise ValueError(f"Invalid dimensions in dim_groups: {dims_bad}")
        dims_uniq, counts = np.unique(dims_flat, return_counts=True)
        repeats = dims_uniq[counts > 1]
        if len(repeats):
            raise ValueError(f"Repeated dimensions in dim_groups: {repeats}")
        dim_groups = [np.array(dim_group, dtype=int) for dim_group in dim_groups]

    var = _init_var(source, target, dim_groups)
    err = np.inf

    transform = np.eye(ndim)
    shift = np.zeros(ndim)

    transformed = source.copy()
    P = np.ones((len(source), len(target)))
    for i in range(max_iters):
        _err, _transform, _shift = err, transform, shift
        transformed = apply_transform(source, transform, shift)
        P = compute_P(transformed, target, var, w)
        transform, shift, var, err = solve_transform(
            source, target, P, dim_groups, var, method
        )
        callback(target, transformed, i, err)

        if _err - err < eps:
            if err > _err:
                transform, shift = _transform, _shift
            break

    return transform, shift, transformed, P


def cpd(
    source: NDArray[np.floating],
    target: NDArray[np.floating],
    w: float = 0.0,
    eps: float = 1e-10,
    max_iters: int = 500,
    callback: Callback = dummy_callback,
    method: str = "affine",
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """
    Compute the CPD.
    This is just a wrapper around joint_cpd that puts everything into one dim_group.
    See the docstring of joint_cpd for details on the parameters and returns.

    See https://arxiv.org/abs/0905.2635 for details on the base CPD algorithm.
    """
    return joint_cpd(
        source=source,
        target=target,
        dim_groups=None,
        w=w,
        eps=eps,
        max_iters=max_iters,
        callback=callback,
        method=method,
    )
