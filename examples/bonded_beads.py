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

class Bead(m.Particle):
    mass = 0.4
    radius = 0.2
    dynamics = m.Overdamped

pot_bb = m.Potential.soft_sphere(kappa=0.2, epsilon=0.05, \
                                 r0=0.2, eta=4, tol=0.01, min=0.01, max=0.5)

# hamonic bond between particles
pot_bond = m.Potential.harmonic(k=0.4, r0=0.2, max = 2)

# angle bond potential
pot_ang = m.Potential.harmonic_angle(k=0.2, theta0 = 0.85*np.pi, tol=0.1)

# bind the potential with the *TYPES* of the particles
m.bind(pot_bb, Bead, Bead)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 0.1)

# bind it just like any other force
m.bind(rforce, Bead)

# make a array of positions
xx = np.arange(4., 16, 0.15)

p = None                              # previous bead
bead = Bead([xx[0], 10., 10.0])       # current bead

for i in range(1, xx.size):
    n = Bead([xx[i], 10.0, 10.0])     # create a new bead particle
    m.Bond(pot_bond, bead, n)             # create a bond between prev and current
    if(i > 1):
        m.Angle(pot_ang, p, bead, n) # make an angle bond between prev, cur, next
    p = bead
    bead = n

# run the simulator interactive
m.Simulator.run()








