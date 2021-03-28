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
m.init(dim=dim,
       cutoff=cutoff,
       clip_planes = [([8, 8, 8], [1, 1, 0]), ([11, 11, 11], [-1, -1, 0])])

class A(m.Particle):
    mass = 40
    radius = 0.4
    dynamics = m.Overdamped


class B(m.Particle):
    mass = 40
    radius = 0.4
    dynamics = m.Overdamped


# create three potentials, for each kind of particle interaction
pot_aa = m.Potential.morse(d=3,   a=5, max=3)
pot_bb = m.Potential.morse(d=3,   a=5, max=3)
pot_ab = m.Potential.morse(d=0.3, a=5, max=3)


# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_aa, A, A)
m.Universe.bind(pot_bb, B, B)
m.Universe.bind(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential

rforce = m.forces.random(0, 50)

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








