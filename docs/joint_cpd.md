# Joint Coherent Point Drift

## Background
Coherent Point Drift (CPD) is a point set registration algorithm originally described in [Myroneko & Song](https://arxiv.org/abs/0905.2635).
This algorithm is an excellent choice for point set registration in cases without known correspondences, outperforming many other state-of-the-art methods.
There are many existing implementations of CPD, such as [pycpd](https://github.com/siavashk/pycpd).

One assumption made by CPD is that your point clouds exist in a space where all the axes form a single basis.
As a result every degree of freedom allowed by your transformation of choice (affine, rigid, etc.) can be used in the final transformation.
This is generally OK in cases where your data is purely spatial, but if your point clouds include a non-spatial axis (ie: color) this can cause issues since transformations that mix the spatial and non-spatial axes can occur.
Joint CPD is an extension of the base CPD algorithm that exists to address this.

## The Joint CPD Algorithm
The joint CPD algorithm is a simple extension to CPD
First we define the concept of "dimensional groups",
where every axis in a given dimensional group is transformed together.
For a $L$-dimensional point cloud we can define $D$ dimensional groups, where $1 \leq D \leq L$. For convenience we define the following notation:

$$
x_{(d)}
$$

Where $x$ is a point cloud, and the $(d)$ denotes that we are referring to the $d$'th dimensional group.
Each dimensional group can contain up to $L$ dimensions, but cannot overlap with each other at all.

With this now defined we can write down the posterior probabilities of the Gaussian Mixture Model (GMM) components as:

$$
P^{old}(m|x_{n}) = \frac{\prod_{d=1}^{D}\exp\left(-\frac{1}{2} \lVert \frac{x_{(d)n} - T_{(d)}(y_{(d)m})}{\sigma_{(d)}} \rVert ^2 \right)}{\sum_{k=1}^{M}\prod_{d=1}^{D}\exp\left(-\frac{1}{2} \lVert \frac{x_{(d)n} - T_{(d)}(y_{(d)k})}{\sigma_{(d)}} \rVert ^2 \right) + \frac{w}{1-w}\frac{M}{N}\prod_{d=1}^{D} \left(2\pi\sigma_{(d)}^2\right)^{l_{(d)}/2}}
$$

Where: 

* $x$ is the target point cloud
* $y$ is the source point cloud (the GMM centroids)
* $N$ is the number of points in $x$
* $M$ is the number of points in $y$
* $D$ is the number of dimensional groups
* $T_{(d)}$ is the transformation in the $d$'th dimensional group
* $\sigma_{(d)}^2$ is the variance of the GMM in the $d$'th dimensional group
* $l_{(d)}$ is the number of dimensions in the $d$'th dimensional group
* $w$ is the weight of the uniform distribution 

And we can write down the objective function as:

$$
Q = -\sum_{n=1}^{N}\sum_{m=1}^{M} P^{old}(m|x_{n})\log\left(P^{new}(m)p^{new}(x_{n}|m)\right)
$$

Because the partial derivatives of $Q$ with respect to the components of $T_{(d)}$ and $\sigma_{(d)}$ kill all terms pertaining to other dimensional groups (see [Proof of Equivalence](joint_cpd.md#proof-of-equivalence)) we can update the transformation and variance as in the original CPD algorithm within each dimensional group. 
This allows us to compute $D$ independent transformations, but because all of the dimensional groups are used to compute $P^{old}$ these transformations can leverage correlations between different dimensional groups.

## Proof of Equivalence

Starting from the objective function $Q$ as defined [above](joint_cpd.md#the-joint-cpd-algorithm) we show that the partial derivatives within each dimensional group reduce to a form that is trivially equivalent to those in the base CPD algorithm below.

First we expand $Q$ as:

$$
Q = -\sum_{n=1}^{N}\sum_{m=1}^{M} P^{old}(m|x_{n})\log\left(\prod_{d=1}^{D} \frac{\exp\left(-\frac{\lVert x_{(d)n} - T_{(d)}(y_{(d)m})\rVert ^2}{2\sigma_{(d)}^2} \right)}{\left(2\pi\sigma_{(d)}^2\right)^{1/2}}\right)
$$

$$
Q = -\sum_{n=1}^{N}\sum_{m=1}^{M} P^{old}(m|x_{n})\sum_{d=1}^{D} -\frac{\lVert x_{(d)n} - T_{(d)}(y_{(d)m})\rVert ^2}{2\sigma_{(d)}^2} - \frac{\log\left(2\pi\sigma_{(d)}^2\right)}{2}
$$

$$
Q = \sum_{d=1}^{D}\frac{N_{P}l_{(d)}}{2}\log\sigma_{(d)}^2 + \sum_{n=1}^{N}\sum_{m=1}^{M} P^{old}(m|x_{n})\sum_{d=1}^{D} \frac{\lVert x_{(d)n} - T_{(d)}(y_{(d)m})\rVert ^2}{2\sigma_{(d)}^2}
$$

Where $N_{P}$ is the sum of $P^{old}$.

Now we take the partial derivatives.
First WLOG take the derivative with respect to $A$ where $A$ is a component of the transform (ie: the affine matrix $B$ in the affine case, the rotation matrix $R$ in the rigid case, etc.):

$$
\frac{dQ}{dA_{(d)}} = \sum_{n=1}^{N}\sum_{m=1}^{M} P^{old}(m|x_{n})\frac{\lVert x_{(d)n} - \frac{dT_{(d)}}{dA_{(d)}}(y_{(d)m})\rVert ^2}{2\sigma_{(d)}^2}
$$

Which is trivially equivalent to the derivative in the base CPD case.

Similarly taking the derivative with respect to $\sigma^2$:

$$
\frac{dQ}{\sigma_{(d)}^2} = \frac{N_{P}l_{(d)}}{2\sigma_{(d)}^2} - \sum_{n=1}^{N}\sum_{m=1}^{M} P^{old}(m|x_{n}) \frac{\lVert x_{(d)n} - T_{(d)}(y_{(d)m})\rVert ^2}{2\sigma_{(d)}^4}
$$

Which is also trivially equivalent to the derivative in the base CPD case.
