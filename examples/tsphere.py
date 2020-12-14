import mechanica as m
import numpy as np

m.Simulator(dim=[25., 25., 25.], dt=0.0005, cutoff=3, bc=m.BOUNDARY_NONE)

class Green(m.Particle):
    mass = 1
    radius = 0.1
    dynamics = m.Overdamped
    style = {'color':'mediumseagreen'}

class Big(m.Particle):
    mass = 10
    radius = 8
    frozen=True
    style = {'color':'orange'}

# simple harmonic potential to pull particles
pot = m.Potential.harmonic(k=1, r0=0.1, max = 3)

# potentials between green and big objects.
pot_yc = m.Potential.glj(e=1, r0=1, m=3, min=0.01)
pot_cc = m.Potential.glj(e=0.0001, r0=0.1, m=3, min=0.005, max=2)

# random points on surface of a sphere
pts = m.random_points(m.Sphere, 10000) * (Green.radius + Big.radius) + m.Universe.center

# make the big particle at the middle
Big(m.Universe.center)

# constuct a particle for each position, make
# a list of particles
beads = [Green(p)  for p in pts]

# create an explicit bond for each pair in the
# list of particles. The bind_pairwise method
# searches for all possible pairs within a cutoff
# distance and connects them with a bond.
#m.bind_pairwise(pot, beads, 0.7)

rforce = m.forces.random(0, 0.01, durration=0.1)

# hook up the potentials
#m.bind(rforce, Green)
m.bind(pot_yc, Big, Green)
m.bind(pot_cc, Green, Green)

m.bind_pairwise(pot, [p for p in m.Universe.particles if p.position[1] < m.Universe.center[1]], 1)

# run the model
m.show()
