.. _clusters-label:



Clusters
========


Clusters and Hierarchical Orgianization
---------------------------------------

A *Cluster* is a special kind of particle that contains other particles,
including more clusters.


Defining Clusters
-----------------

We can create an instance of a cluster simply via...


Clusters are most usefull when they contain nested particle types, We define
cluster with embedded types with standard python syntax::

  import mechanica as m

  class MyCell(m.Cluster):
    class A(m.Particle)
    class B(m.Particle)

This defines a new cluster type that contains two nested particle types, `A` and
`B`.

Creating Clusters
-----------------

Simply::

  c = MyCell()

  # creates an instance of the MyCell.A particle type in the cluster.
  c.A()

  # same for B
  c.B()


Defining Interactions
---------------------

Create a potential just like for free particles::

  pot = m.Potential.soft_sphere(kappa=20, epsilon=5.0, \
                                r0=0.2, eta=4, tol=0.01, min=0.01, max=0.5)

We allow interactions between particles that belong to a cluster, and between
particles belonging to different clusters, use the `bound` argument to the
`bind` fuction::

  m.Universe.bind(pot, MyCell.A, MyCell.A, bound=True)

Or to create an interaction potential that is between ALL instances of the
particle type, reguardless of wether or not it is in cluster, simply set bound
to false::
  m.Universe.bind(pot, MyCell.A, MyCell.A, bound=False)




Stuff
^^^^^





