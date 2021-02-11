Advanced Epiboly With Clusters and Events
-----------------------------------------

We can write a much more detailed model of epiboly using clusters and events::

  import mechanica as m
  import numpy as np

  # dimensions of universe
  dim=np.array([30., 30., 30.])
  center = dim / 2

  m.Simulator(
    dim=dim,
    cutoff=12,
    integrator=m.FORWARD_EULER,
    dt=0.0005)

  class C(m.Cluster):
    radius=2.3

    class B(m.Particle):
      radius=0.25
      dynamics = m.Overdamped
      mass=15
      style={"color":"skyblue"}

    def split(self, event):
      print("C.split(" + str(self) + ", event: " + str(event) + ")")
      axis = self.position - C.yolk_pos
      print("axis: " + str(axis))
      m.Cluster.split(self, axis=axis)

    events = [m.on_time(split,
                        period=0.2,
                        predicate="largest")]

  class Yolk(m.Particle):
    radius = 10
    mass = 1000000
    dynamics=m.Overdamped
    style={"color":"gold"}

  total_height = 2 * Yolk.radius + 2 * C.radius
  yshift = total_height/2 - Yolk.radius
  cshift = total_height/2 - C.radius - 1

  yolk = Yolk(position=center-[0., 0., yshift])

  c = C(position=center+[0., 0., cshift])

  C.yolk_pos = yolk.position

  c.B(4000)

  pb  = m.Potential.soft_sphere(kappa=300, epsilon=6, r0=0.5, \
                                eta=2, tol = 0.05, min=0, max=3)

  pub = m.Potential.soft_sphere(kappa=400, epsilon=0, r0=0.5, \
                                eta=2, tol = 0.05, min=0.001, max=1.5)

  py = m.Potential.soft_sphere(kappa=300, epsilon=25, r0=1, \
                               eta=2, tol = 0.04, min=0, \
                               max=10, shift=True)

  rforce = m.forces.random(0, 1)

  m.bind(rforce, C.B)
  m.bind(pb, C.B, C.B, bound=True)
  m.bind(pub, C.B, C.B, bound=False)
  m.bind(py, Yolk, C.B)

  m.Simulator.irun()




The complete simulation script is here, and can be downloaded here:

Download: :download:`this example script <../../examples/epiboly_cluster.py>`::
