import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

count = 3000

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff)

class Big(m.Particle):
    mass = 500000
    radius = 3

class Small(m.Particle):
    mass = 0.1
    radius = 0.2
    target_temperature=0


pot_bs = m.Potential.soft_sphere(kappa=100, epsilon=1, r0=3.2, \
    eta=3, tol = 0.1, min=0.1, max=8)

pot_ss = m.Potential.soft_sphere(kappa=1, epsilon=0.1, r0=0.2, \
    eta=2, tol = 0.05, min=0.01, max=4)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_bs, Big, Small)
m.Universe.bind(pot_ss, Small, Small)

Big(position=center, velocity=[0., 0., 0.])

for p in m.random_points(m.Disk, count) * \
    2.5 * Big.radius + center + [0, 0, Big.radius + 1]:
    Small(p)

# run the simulator interactive
m.Simulator.run()








