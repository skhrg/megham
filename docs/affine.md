# Computing and Decomposing the Affine Transformation

## The Affine Transformation

The affine transformation is a supremely useful transformation that preserves collinearity and parallelism but nothing else.
This is useful in many scenarios where you which to remain in the linear regime while allowing as generic of a transformation as possible.
There are many places to learn the details of the transformation, but as usual [Wikipedia](https://en.wikipedia.org/wiki/Affine_transformation) is a good starting place. 

This repository contains a function, [`get_affine`](https://skhrg.github.io/megham/latest/reference/transform/#megham.transform.get_affine), to compute this transform.
This page contains information on the two methods used in that function to compute this transformation.


Since the affine transformation is simply a generic linear transformation, we write it as follows:

$$
y = Ax + b
$$

Where $x$ is the starting set of points, $A$ is a matrix representing the affine transformation, $b$ is a offset vector, and $y$ is the transformed set of points.
We will refer to the terms of this equation throughout this page.

## Method 1: The SVD Method

Using the SVD to compute the affine transform is incredibly useful when dealing with noisy or large datasets (or large noisy datasets).

To begin we first approximate $b$ by taking the mean or median of $y - x$.
Then we build the following matrix:

$$
M = \begin{bmatrix} x^T & y^T \end{bmatrix}
$$

we then use the SVD to compute $V^{\ast}$, the right singular vectors of $M$.

Taking the two largest right singular vectors, $V_{1}$ and $V_{2}$ we define the following:

$$
\begin{bmatrix} B \\ C \end{bmatrix} = \begin{bmatrix} V_{1} & V_{2} \end{bmatrix}
$$

which then gives us the affine transformation $A = CB^{-1}$.

This may seem like nonsense at first but its actually fairly intuitive method.
$B$ essentially is a description of $x$'s column basis and $C$ the same for $y$,
so $CB^{^1}$ takes us from the basis of $x$ to the singular basis and then to the basis of $y$.
Given that $x$ and $y$ are $n_{point}$ by $n_{dim}$ matrices, the column basis describes the coordinate systems in which they live.

## Method 2: Weighted Least Squares

This method is much more straightforward to understand.
In the original equation $y = Ax + b$ we add an additional variable $w$ where $w$ is a per-point weight.
The equation now becomes:

$$
\sqrt{2}y = A\sqrt{2}(x + b)
$$

In order to simplify this equation we split $b$ into two terms: $b_{naive}$ and $b_{leastsq}$.
We solve for $b_{naive}$ simply by taking the mean or median of $y - b$ just as before.

Then to solve for $b_{leastsq}$ we define the following:

$$
A' = \begin{bmatrix}A\\ b_{leastsq} \end{bmatrix}
$$

and

$$
x' = \begin{bmatrix}\sqrt{w}x & \mathbf{1} \end{bmatrix}
$$

where $\mathbf{1}$ is a vector of all ones.

Now we have:

$$
\sqrt{w}\left(y - b_{naive}\right) = A'x'
$$

from which we can solve for $A'$.

The analytic solution is simply:

$$
A' = \sqrt{w}yx'^{-1}
$$

but direct inversion of $x'$ can be costly if it is a large array and can have numerical issues
if it has a poor condition number, so instead we use the standard [numpy function](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) to solve.

Then we can extract $A$ and $b_{leastsq}$ from $A'$ and sum up to get the full $b = b_{naive} + b_{leastsq}$.

## Decomposing the Affine Transformation

Regardless of how we compute the affine transformation we often want to decompose it into a set of numbers which can then be understood.
The function [`decompose_affine`](https://skhrg.github.io/megham/latest/reference/transform/#megham.transform.decompose_affine) decomposes the affine transform into a scale term, a shear term, and a rotation matrix.
This decomposition is able to fully describe the transformation and offers values that are easy to interpret geometrically.

We define this decomposition as:

$$
A = RSM
$$

where $R$ is the rotation matrix, $S$ is the shear matrix, and $M$ is the scale matrix.

Starting with the affine transform as a $n_{dim}$ by $n_{dim}$ matrix we take:

$$
B = A^T A = M^T S^T R^T R S M = M^T S^T S M
$$

$B$ is now the transform without the rotation term since the transpose of a rotation matrix is the inverse of said rotation matrix.

We can then use a Cholesky decomposition to extract

$$
C = SM
$$

from $B$.

Since the scale matrix is fully diagonal and the diagonal of a shear matrix is the identity we can compute them as follows:

$$
M = diag \left( C \right)
$$

$$
S = CM^{-1}
$$

Now to get the rotation matrix we can simply take:

$$
R = AC^{-1} = R S M \left( SM \right) ^{-1}
$$

The rotation matrix can then be converted to a something like an euler angle sequence for further intuition.
