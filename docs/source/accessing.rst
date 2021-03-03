Finding and Accessing Objects
=============================
One of the more comming user activities is finding, accessing and interacting
with system objects after we've created them.


Each :any:`Particle` derived type has an :func:`Particle.items()` method on the
type that returns all of the objects of that type::

  class MyType(m.Particle):
      ...

  # make a few instances...
  MyType()

  # get all the instances of that type:

  parts = MyType.items()

We can acess ALL of the particles in the entire simulation via the
:any:`universe.particles()` function.


Frequently we might want to grab all the objects in a grid, for example, if we
want to display some quantity as a function of spatial position. We can use the
:any:`universe.grid()` function to get all the particles binned in on a regular
grid. Each element in the returned 3D array is a particle list of the particles
at that grid site. For example, if we wanted a 10x10x10 grid, of particles, we
would::

  parts = universe.grid([10, 10, 10])

  
          






Finding Neighbors
-----------------

Each object spatial object is aware if it's neighbors, and we can get a list of
all the neghbors of an object by calling the :any:`Particle.neighbors` method.

The `neighbors` method accepts two optional arguments, `distance` and
`types`. Distance is the distance away from the *surface* of the present
particle to search for neibhbors, and types is a tuple of particle types to
restrict the search to. For example, to search for all objects of type `A` and
`B` a distance of 1 unit away from a particle `p`, we would::

  >>> nbrs = p.neighbors(distance=1, types=(A, B))
  >>> print(len(nbrs))
  



