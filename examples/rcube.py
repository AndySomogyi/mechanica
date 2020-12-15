import mechanica as m
import numpy as np

m.Simulator(dim=[20., 20., 20.], cutoff=8, bc=m.BOUNDARY_NONE)

class Bead(m.Particle):
    mass = 1
    radius = 0.1
    dynamics = m.Overdamped


class Blue(m.Particle):
    mass = 1
    radius = 0.1
    dynamics = m.Overdamped
    style = {'color':'blue'}


pot = m.Potential.harmonic(k=1, r0=0.1, max = 3)

pts = m.random_points(m.SolidCube, 10000) * 18 + m.Universe.center

beads = [Bead(p) for p in pts]

m.show()
