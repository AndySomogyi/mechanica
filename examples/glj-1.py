import mechanica as m
import numpy as np

cutoff = 8
count = 3

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

m.Simulator(dim=dim, cutoff=cutoff)

class B(m.Particle):
    mass = 1
    dynamics = m.Overdamped

# make a glj potential, this automatically reads the
# particle radius to determine rest distance.
pot = m.Potential.glj(e=1)

m.bind(pot, B, B)

p1 = B(center + (-2, 0, 0))
p2 = B(center + (2, 0, 0))
p1.radius = 1
p2.radius = 2

m.Simulator.run()
