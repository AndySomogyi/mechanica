import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

receptor_count = 500

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff, cells=[4, 4, 4], integrator=m.RUNGE_KUTTA_4)

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
receptor_pts = m.random_points(m.Sphere, receptor_count) * Big.radius + center

pot_rr = m.Potential.soft_sphere(kappa=0.02,    epsilon=0, r0=0.5, eta=2, tol=0.05, min=0.01, max=4)
pot_vr = m.Potential.soft_sphere(kappa=0.02,    epsilon=0.1, r0=0.6, eta=4, tol=0.05, min=0.01, max=3)
pot_vb = m.Potential.soft_sphere(kappa=5,  epsilon=0, r0=4.8, eta=4, tol=0.05, min=3, max=5.5)

# bind the potential with the *TYPES* of the particles
m.bind(pot_rr, Receptor, Receptor)
m.bind(pot_vr, Receptor, Virus)
m.bind(pot_vb, Big, Virus)

# create a random force (Brownian motion), zero mean of given amplitide
tstat = m.forces.random(0, 0.1)
vtstat = m.forces.random(0, 0.1)

# bind it just like any other force
m.bind(tstat, Receptor)

m.bind(vtstat, Virus)

b=Big(position=center, velocity=[0., 0., 0.])

Virus(position=center+[0, 0, Big.radius + 0.75])

harmonic = m.Potential.harmonic(k=0.01*500, r0=Big.radius)

for p in receptor_pts:
    r=Receptor(p)
    m.bind(harmonic, b, r)

# run the simulator interactive
m.Simulator.run()
