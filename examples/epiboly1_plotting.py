import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 0.5

count = 3000

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff)

class Yolk(m.Particle):
    mass = 500000
    radius = 3

class Cell(m.Particle):
    mass = 5
    radius = 0.2
    target_temperature=0
    dynamics = m.Overdamped

pot_bs = m.Potential.soft_sphere(kappa=5, epsilon=20, r0=2.9, \
    eta=3, tol = 0.1, min=0, max=9)
pot_ss = m.Potential.soft_sphere(kappa=10, epsilon=0.000000001, r0=0.2, \
    eta=2, tol = 0.05, min=0, max=3)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_bs, Yolk, Cell)
m.Universe.bind(pot_ss, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 0.05)

# bind it just like any other force
m.bind(rforce, Cell)

yolk = Yolk(position=center, velocity=[0., 0., 0.])

import sphericalplot as sp

for p in m.random_points(m.SolidSphere, count) * \
    0.5 * Yolk.radius + center + [0, 0, 1.3 * Yolk.radius]:
    Cell(p)


plt = sp.SphericalPlot(Cell.items(), yolk.position)

m.on_time(plt.update, period=0.01)

# run the simulator interactive
m.Simulator.run()








