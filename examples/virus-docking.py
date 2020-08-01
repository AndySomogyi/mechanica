import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

receptor_count = 500

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator, don't load any example
m.Simulator(example="", dim=dim, cutoff=cutoff, cells=[4, 4, 4])

class Big(m.Particle):
    mass = 500000
    radius = 3

class Receptor(m.Particle):
    mass = 0.1
    radius = 0.1
    target_temperature=1
    dynamics = m.Overdamped

class Virus(m.Particle):
    mass = 1
    radius = 0.5
    target_temperature=1
    dynamics = m.Overdamped

# locations of initial receptor positions
receptor_pts = m.random_point(m.Sphere, receptor_count) * Big.radius + center

pot_bs = m.Potential.soft_sphere(kappa=10, epsilon=50, r0=3, eta=3, tol = 0.1, min=0.1, max=9)
pot_rr = m.Potential.soft_sphere(kappa=2, epsilon=0, r0=0.5, eta=2, tol=0.05, min=0.01, max=4)
pot_vr = m.Potential.soft_sphere(kappa=2, epsilon=10, r0=0.6, eta=4, tol=0.05, min=0.01, max=1)
pot_vb = m.Potential.soft_sphere(kappa=450, epsilon=0, r0=4.8, eta=4, tol=0.05, min=3, max=5.5)

# bind the potential with the *TYPES* of the particles
m.bind(pot_rr, Receptor, Receptor)
m.bind(pot_vr, Receptor, Virus)
m.bind(pot_vb, Big, Virus)

# create a random force (Brownian motion), zero mean of given amplitide
tstat = m.forces.random(0, 3)
vtstat = m.forces.random(0, 5)

# bind it just like any other force
m.bind(tstat, Receptor)

m.bind(vtstat, Virus)

b=Big(position=center, velocity=[0., 0., 0.])

Virus(position=center+[0, 0, Big.radius + 0.75])

harmonic = m.Potential.harmonic(K=500, r0=Big.radius)

for p in receptor_pts:
    r=Receptor(p)
    m.bind(harmonic, b, r)

# run the simulator interactive
m.Simulator.run()
