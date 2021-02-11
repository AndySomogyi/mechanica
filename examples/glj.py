import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

count = 3

# dimensions of universe
dim=np.array([20., 20., 20.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff)

class Bead(m.Particle):
    mass = 1
    radius = 0.5
    dynamics = m.Overdamped

pot = m.Potential.glj(e=1)

# bind the potential with the *TYPES* of the particles
m.bind(pot, Bead, Bead)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 0.01)
m.bind(rforce, Bead)


r = 0.8 * Bead.radius

positions = [center + (x, 0, 0) for x in
             np.arange(-count*r + r, count*r + r, 2*r)]



for p in positions:
    print("position: ", p)
    Bead(p)

# run the simulator interactive
m.Simulator.run()








