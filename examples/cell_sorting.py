import mechanica as m
import numpy as np

# total number of cells
A_count = 5000
B_count = 5000

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff)

class A(m.Particle):
    mass = 5
    radius = 0.5
    dynamics = m.Overdamped


class B(m.Particle):
    mass = 5
    radius = 0.5
    dynamics = m.Overdamped


# create three potentials, for each kind of particle interaction
pot_aa = m.Potential.soft_sphere(kappa=5, epsilon=0.25, r0=1, \
                                 eta=2, tol = 0.05, min=0.01, max=3)

pot_bb = m.Potential.soft_sphere(kappa=5, epsilon=0.25, r0=1, \
                                 eta=2, tol = 0.05, min=0.01, max=3)

pot_ab = m.Potential.soft_sphere(kappa=5, epsilon=0.0025, r0=1, \
                                 eta=2, tol = 0.05, min=0.01, max=3)


# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_aa, A, A)
m.Universe.bind(pot_bb, B, B)
m.Universe.bind(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential

rforce = m.forces.random(0, 5)

# bind it just like any other force
m.bind(rforce, A)
m.bind(rforce, B)

# create particle instances, for a total A_count + B_count cells
for p in np.random.random((A_count,3)) * 15 + 2.5:
    A(p)

for p in np.random.random((B_count,3)) * 15 + 2.5:
    B(p)

# run the simulator
m.Simulator.run()








