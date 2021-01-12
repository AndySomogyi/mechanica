Metrics and Derived Quantities
==============================

Mechanica provides numerous methods to compute a range of derived
quantities. Some of these are top-level metrics, and depend on the entire
simulation volume, and others are localized to individual objects. :any:`Simulator`


Pressure and Virial Tensors
---------------------------
For a system of N particles in a volume V, we can compute the  surface tension
from the diagonal components of the pressure tensor
:math:`P_{\alpha,\alpha}(\alpha=x,y,z)`. The :math:`P_{xx}` components are:

.. math::

   P_{\alpha,\beta} = \rho k T + \
       \frac{1}{V} \
       \left( \
       \sum^{N-1}_{i=1} \
       \sum^{N}_{j>i} \
       (\mathbf{r}_{ij})_{\alpha} \
       (\mathbf{f}_{ij})_{\beta} \
       \right),

where :math:`N` is the number of particles, :math:`\rho` is the particle density
density, :math:`k` Boltzmann constant, :math:`T` is the temperature,
:math:`\mathbf{r}_{ij}` is the vector between the particles :math:`i` and :math:`j`,
and :math:`\mathbf{f}_{ij}` is the force between them. Some important concepts
here is that the force here is *only* between the particles used in the pressure
tensor calculation, it specifically excludes any external force. The pressure
tensor here is a measure of how much *internal* force exists in the specified
set of particles.

.. _virial:

We comonly refer to the right hand side above as the `virial`, it represents
half of the the product of the stress due to the net force between pairs of
particles and the distance between them. We formally define the virial tensor
components as

.. math::
   V_{\alpha,\beta} = \sum^{N-1}_{i=1} \
       \sum^{N}_{j>i} \
       (\mathbf{r}_{ij})_{\alpha} \
       (\mathbf{f}_{ij})_{\beta}.
   :label: eqn-virial


The volume of a group of particles is not well defined, as such we separate out
computing the virial component, and the volume, and give users the flexiblity of
using different volume metrics. 

.. _my-reference-label:

We provide a number of different options for calculating the virial
tensor. You can compute the pressure tensor for the entire simulation domain, or
a specific region using the :meth:`Universe.virial` method. Can compute the
pressure tensor for a specific cluster using the :meth:`Cluster.virial` method,
or can compute the tensor at a specific particle location using
:meth:`Particle.virial` method. 




Radius of Gyration
------------------


In the radius of gyration is measure of the dimensions of a group
(:any:`Cluster`) of particles such as a polymer chain, macro-molecule or some
larger object.  The radius of gyration of group of particles at a given time is
defined as:

.. math:: 
   R_\mathrm{g}^2 \ \stackrel{\mathrm{def}}{=}\ 
   \frac{1}{N} \sum_{k=1}^{N} \left( \mathbf{r}_k - \mathbf{r}_\mathrm{mean}
   \right)^2

We can compute the radius of gyration for a cluster of particles using the
:meth:`Cluster.radius_of_gyration` method. 



Center of Mass
--------------

The center of mass of a system of particles, :math:`P_i, i-1, \ldits, n`, each
with mass :math:`m_i`, at locations :math:`\mathbf{r}_i, i=1, \ldots, n`, with
the center of mass :math:`\mathbf{R}` satisfy the condition

.. math::

   \sum_{i=1}^n m_i(\mathbf{r}_i - \mathbf{R}) = \mathbf{0},

with :math:`\mathbf{R}` defined as:

.. math::

   \mathbf{R} = \frac{1}{M} \sum_{i=1}^n m_i \mathbf{r}_i,

where :math:`M` is the sum of the masses of all of the particles.

We can compute the center of mass of a cluster particles with the
:meth:`Cluster.center_of_mass` method. 


Center of Geometry
------------------

Computes the geometric center of a group of particles with the
:meth:`Cluster.center_of_geometry` method, or equivalently, with the
:meth:`Cluster.centroid` method. 


Moment of Inertia
-----------------

For a system of :math:`N` particles, the moment of inertia tensor is a symmetric
tensor, and is defined as:

.. math::
   \mathbf{I} =
   \begin{bmatrix}
   I_{11} & I_{12} & I_{13} \\
   I_{21} & I_{22} & I_{23} \\
   I_{31} & I_{32} & I_{33}
   \end{bmatrix}

Its diagonal elements are defined as

.. math::

   \begin{align}
   I_{xx} \stackrel{\mathrm{def}}{=}  \sum_{k=1}^{N} m_{k} (y_{k}^{2}+z_{k}^{2}), \\
   I_{yy} \stackrel{\mathrm{def}}{=}  \sum_{k=1}^{N} m_{k} (x_{k}^{2}+z_{k}^{2}), \\
   I_{zz} \stackrel{\mathrm{def}}{=}  \sum_{k=1}^{N} m_{k} (x_{k}^{2}+y_{k}^{2})
   \end{align}


and the  the off-diagonal elements, also called the are:

.. math::
   \begin{align}
   I_{xy} = I_{yx} \ \stackrel{\mathrm{def}}{=}\  -\sum_{k=1}^{N} m_{k} x_{k} y_{k}, \\ 
   I_{xz} = I_{zx} \ \stackrel{\mathrm{def}}{=}\  -\sum_{k=1}^{N} m_{k} x_{k} z_{k}, \\
   I_{yz} = I_{zy} \ \stackrel{\mathrm{def}}{=}\  -\sum_{k=1}^{N} m_{k} y_{k} z_{k}
   \end{align}

We can compute the inertia tensor for a group of particles using the
:meth:`Cluster.moment_of_inertia` or :meth:`Cluster.inertia` methods. 






