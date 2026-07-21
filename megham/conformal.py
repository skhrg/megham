r"""
Construct a conformal transformation between two
point cloud using Lie algebra.

This uses the canonical basis of $SO(n+1,1)$
in the null basis used by conformal geometry.

This code mostly follows the method for this outlined in
Optimization Algorithms on Matrix Manifolds (Absil, Mahony, and Sepulchre).
But the following was also used for reference:

* Lie Groups, Lie Algebras, and Representations (Hall)
* Geometric Algebra for Computer Science (Dorst, Fontijne, and Mann)

(all hail the JHU math and CS departments for making me download PDFs of these so long ago)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm


def conformal_metric(n: int) -> NDArray:
    r"""
    Return the metric matrix in the null basis.

    The conformal group acts on $R^{(n+1,1)}$.

    In the null basis we have:
    
    $$
    \eta = 
    \begin{matrix}
    I_n & 0 & 0\\
    0 & 0 & 1 \\
    0 & 1 & 0 \\
    \end{matrix}
    $$

    and the Lie algebra condition is

    $X^T \eta + \eta @ X = 0$.

    Parameters
    ----------
    n : int
        Dimension of Euclidean space.

    Returns
    -------
    eta : NDArray[np.floating]
        Array of shape `(n+2, n+2)`
        with the metric in the null basis
    """
    eta = np.zeros((n + 2, n + 2))

    eta[:n, :n] = np.eye(n)

    eta[n, n + 1] = 1
    eta[n + 1, n] = 1

    return eta


def rotation_generators(n: int) -> list[NDArray[np.floating]]:
    r"""
    Generate rotation generators $_{ij}$.

    Parameters
    ----------
    n : int
        Dimension of Euclidean space.

    Returns
    -------
    generators : list[NDArray[np.floating]]
        List of rotation generators.
    """

    generators = []
    for i in range(n):
        for j in range(i + 1, n):
            R = np.zeros((n + 2, n + 2))
            R[i, j] = 1
            R[j, i] = -1
            generators += [R]

    return generators


def translation_generators(n: int) -> list[NDArray[np.floating]]:
    r"""
    Generate translation generators $B_i$.

    Parameters
    ----------
    n : int
        Dimension of Euclidean space.

    Returns
    -------
    generators : list[NDArray[np.floating]]
        List of translaion generators.
    """
    generators = []
    for i in range(n):
        B = np.zeros((n + 2, n + 2))
        B[i, n] = 1
        B[n + 1, i] = -1
        generators += [B]

    return generators


def scale_generators(n: int) -> list[NDArray[np.floating]]:
    r"""
    Generate the dilation generator $S$.

    Parameters
    ----------
    n : int
        Dimension of Euclidean space.

    Returns
    -------
    generator : list[NDArray[np.floating]]
        The scale generators.
    """
    S = np.zeros((n + 2, n + 2))
    S[n, n] = 1
    S[n + 1, n + 1] = -1

    return [
        S,
    ]


def conformal_generators(n: int) -> list[NDArray[np.floating]]:
    r"""
    Generate conformal generators $K_i$.
    These correspond to infinitesimal transformations:

    $$
    x -> x + b \|x\|^2 - 2(bx)x
    $$

    Parameters
    ----------
    n : int
        Dimension of Euclidean space.

    Returns
    -------
    generators : list[NDArray[np.floating]]
        The conformal generators.
    """

    generators = []

    for i in range(n):
        K = np.zeros((n + 2, n + 2))
        K[i, n + 1] = 1
        K[n, i] = -1
        generators += K

    return generators


def conformal_basis(n: int) -> list[NDArray[np.floating]]:
    r"""
    Construct the full canonical basis of the conformal Lie algebra.

    Parameters
    ----------
    n : int
        Euclidean dimension.

    Returns
    -------
    basis : list[NDArray[np.floating]]
        The full basis with $(n+1)(n+2)/2$ generators.
        The generators are in the order of:

        * rotation
        * translation
        * scale
        * conformal
    """
    basis = (
        rotation_generators(n)
        + translation_generators(n)
        + scale_generators(n)
        + conformal_generators(n)
    )

    return basis


def conformal_matrix(
    theta: NDArray[np.floating], basis: list[NDArray[np.floating]]
) -> NDArray[np.floating]:
    r"""
    Convert Lie algebra coordinates into a group element.

    Parameters
    ----------
    theta : NDArray[np.floating]
        Lie algebra coordinates.
    basis : list[NDArray[np.floating]]
        Conformal Lie algebra basis.
        Should be the output of `conformal_generators`.

    Returns
    -------
    G : ndarray
        Element of $SO(n+1,1)$ for the input basis.
        Note that this uses the exponential map.
    """
    A = np.zeros_like(basis[0])
    for t, B in zip(theta, basis):
        A += t * B
    return expm(A)


def embed_points(x: NDArray[np.floating]) -> NDArray[np.floating]:
    r"""
    Embed Euclidean points into the conformal light cone.
    The embedding used is:

    $$
    X(x)=
    \begin{bmatrix}
    x \\
    \frac{1-||x||^2}{2} \\
    \frac{1+||x||^2}{2}
    \end{bmatrix}.
    $$

    This then satisfies

    $$
    X^T \eta X = 0
    $$

    meaning all Euclidean points lie on the null cone of
    $R^{(n+1,1)}$.

    Parameters
    ----------
    x : NDArray[np.floating]
        Points in Euclidean space $R^n$
        with shape `(npoint, ndim)`.

    Returns
    -------
    X : NDArray[np.floating]
        Homogeneous conformal coordinates.
        Has shape `(npoint, ndim + 2)`.
    """
    r2 = np.sum(x * x, axis=1, keepdims=True)
    return np.concatenate([x, (1 - r2) / 2, (1 + r2) / 2], axis=1)


def project_points(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Project conformal coordinates back to Euclidean space.

    Parameters
    ----------
    X : NDArray[np.floating]
        Homogeneous conformal coordinates.
        Has shape `(npoint, ndim + 2)`.

    Returns
    -------
    x : NDArray[np.floating]
        Points in Euclidean space $R^n$
        with shape `(npoint, ndim)`.
    """
    u = X[:, :-2]
    denom = X[:, -1] - X[:, -2]  # scale from projection crap
    return u / denom[:, None]


def apply_conformal_transform(x: NDArray[np.floating], G: NDArray[np.floating]):
    """
    Apply a conformal transformation to Euclidean points.
    The transformation is given by:

    $$
    x
    \rightarrow
    X(x)
    \rightarrow
    G X(x)
    \rightarrow
    x'.
    $$

    Parameters
    ----------
    x : NDArray[np.floating]
        Points in Euclidean space $R^n$
        with shape `(npoint, ndim)`.
    G : NDArray[np.floating]
        The conformal matrix.

    Returns
    -------
    y : NDArray[np.floating]
        Transformed oints in Euclidean space $R^n$
        with shape `(npoint, ndim)`.
    """
    X = embed_points(x)
    Y = (G @ X.T).T

    return project_points(Y)


def project_jacobian(
    X: NDArray[np.floating], dX: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute the differential of the conformal projection.

    Parameters
    ----------
    X : NDArray[np.floating]
        Homogeneous conformal coordinates.
        Has shape `(npoint, ndim + 2)`.
    dX : NDArray[np.floating]
        Differential of X.
        Has shape `(npoint, ndim + 2)`.

    Returns
    -------
    dx : NDArray[np.floating]
        The differential.
    """

    u = X[:, :-2]
    a = X[:, -2]
    b = X[:, -1]
    w = b - a

    du = dX[:, :-2]
    da = dX[:, -2]
    db = dX[:, -1]

    dw = db - da

    return du / w[:, None] - u * dw[:, None] / (w * w)[:, None]


def conformal_jacobian(
    G: NDArray[np.floating], x: NDArray[np.floating], basis: list[NDArray[np.floating]]
):
    r"""
    Compute the analytic Lie-group Jacobian.

    WLOG to compute this for some generator $B_i$ we take:

    $$
    \frac{\partial Gx}{\partial \theta_i} = B_i GX .
    $$

    and then apply the conformal projection.

    Parameters
    ----------
    G : NDArray[np.floating]
        The conformal matrix.
    x : NDArray[np.floating]
        Points in Euclidean space $R^n$
        with shape `(npoint, ndim)`.
    basis : list[NDArray[np.floating]]
        Conformal Lie algebra generators.

    Returns
    -------
    J : NDArray[np.floating]
     `(ndim*npoint, nbasis)` array
     of the Jacobian.
    """

    X = embed_points(x)
    Y = (G @ X.T).T
    J = np.zeros((np.prod(x.shape), len(basis)))
    for j, B in enumerate(basis):
        dY = (B @ Y.T).T
        dx = project_jacobian(Y, dY)
        J[:, j] = dx.ravel()

    return J


def get_conformal(
    src: NDArray[np.floating],
    dst: NDArray[np.floating],
    max_iters: int = 10,
    epsilon: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Register the conformal trasnform of two corresponding point clouds
    with a Gauss-Newton type thing.

    Parameters
    ----------
    src : NDArray[np.floating]
        A (npoints, ndim) array of source points.
    dst : NDArray[np.floating]
        A (npoints, ndim) array of destination points.
    max_iters : int, default: 10
        The number of iterations to run the optimizer for.
    epsilon : float, default: 1e-10
        If the change in error drops below this terminate early.

    Returns
    -------
    G : NDArray[np.floating]
        The fit conformal matrix.
    """
    if src.shape != dst.shape:
        raise ValueError("Input point clouds should have the same shape")

    msk = np.isfinite(src).all(axis=1) * np.isfinite(dst).all(axis=1)
    npoint, ndim = src.shape
    if np.sum(msk) < ndim * (ndim - 1) / 2:
        raise ValueError("Not enough finite points to compute transformation")
    if ndim < 3:
        raise ValueError("Need at least 2 spatial dimensions to compute transform")

    basis = conformal_basis(ndim)
    G = np.eye(basis[0].shape[0])
    srx_X = embed_points(src[msk])
    for _ in range(max_iters):
        src_Y = (G @ srx_X.T).T
        src_y = project_points(src_Y)
        residual = (src_y - dst[msk]).ravel()
        J = conformal_jacobian(G, src, basis)
        delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)
        dA = np.zeros_like(G)
        for d, B in zip(delta, basis):
            dA += d * B
        G = expm(dA) @ G
        if np.linalg.norm(delta) < epsilon:
            break
    return G
