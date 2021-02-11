import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

receptor_count = 10000

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff, cells=[4, 4, 4], threads=8)

class Nucleus(m.Particle):
    mass = 500000
    radius = 1

class Receptor(m.Particle):
    mass = 0.2
    radius = 0.05
    target_temperature=1
    #dynamics = m.Overdamped

# locations of initial receptor positions
receptor_pts = m.random_points(m.SolidSphere, receptor_count) * 5  + center

pot_nr = m.Potential.well(k=15, n=3, r0=7)
pot_rr = m.Potential.soft_sphere(kappa=15, epsilon=0, r0=0.3, eta=2, tol=0.05, min=0.01, max=1)

# bind the potential with the *TYPES* of the particles
m.bind(pot_rr, Receptor, Receptor)
m.bind(pot_nr, Nucleus, Receptor)

# create a random force (Brownian motion), zero mean of given amplitide
tstat = m.forces.random(0, 3)
vtstat = m.forces.random(0, 5)

# bind it just like any other force
m.bind(tstat, Receptor)


n=Nucleus(position=center, velocity=[0., 0., 0.])

for p in receptor_pts:
    Receptor(p)

# run the simulator interactive
m.Simulator.run()
