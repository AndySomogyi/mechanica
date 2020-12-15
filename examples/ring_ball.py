import mechanica as m
import numpy as np

m.Simulator(dim=[20., 20., 20.], cutoff=8, bc=m.BOUNDARY_NONE)

class Bead(m.Particle):
    mass = 1
    radius = 0.1
    dynamics = m.Overdamped

# simple harmonic potential to pull particles
pot = m.Potential.harmonic(k=1, r0=0.1, max = 3)

# make a ring of of 50 particles
pts = m.points(m.Ring, 50) * 5 + m.Universe.center

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p) for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
m.bind_pairwise(pot, beads, 1)

# run the model
m.run()
