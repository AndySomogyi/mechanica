import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

count = 3000

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator, don't load any example
m.Simulator(example="", dim=dim, cutoff=cutoff)

class Big(m.Particle):
    mass = 500000
    radius = 3

class Small(m.Particle):
    mass = 0.1
    radius = 0.2
    target_temperature=0
    dynamics = m.Overdamped

pot_bs = m.Potential.soft_sphere(kappa=10, epsilon=50, r0=2.9, \
    eta=3, tol = 0.1, min=0.1, max=9)
pot_ss = m.Potential.soft_sphere(kappa=20, epsilon=0.0001, r0=0.2, \
    eta=2, tol = 0.05, min=0.01, max=3)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_bs, Big, Small)
m.Universe.bind(pot_ss, Small, Small)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 0.00000001)

# bind it just like any other force
m.bind(rforce, Small)

Big(position=center, velocity=[0., 0., 0.])

for p in m.random_point(m.Disk, count) * \
    1.5 * Big.radius + center + [0, 0, Big.radius + 1]:
    Small(p)

# run the simulator interactive
m.Simulator.run()








