import mechanica as m
import numpy as np

m.Simulator(dt=0.1, dim=[15, 12, 10],
            bc={'x':'no_slip',
                'y':'periodic',
                'bottom':'no_slip',
                'top':{'velocity':[-1, 0, 0]}},
            perfcounter_period=100)

# lattice spacing
a = 0.3

#m.universe.boundary_conditions.left.restore = 0.5

class A (m.Particle):
    radius = 0.2
    style={"color":"seagreen"}
    dynamics = m.Newtonian
    mass=10

dpd = m.Potential.dpd(alpha=0.5, gamma=0.1, sigma=1, cutoff=0.5)
dpd_wall = m.Potential.dpd(alpha=0.5, gamma=10, sigma=1, cutoff=0.1)
dpd_left = m.Potential.dpd(alpha=0.5, gamma=100, sigma=1, cutoff=0.5)

m.bind(dpd, A, A)
m.bind(dpd_wall, A, m.Universe.boundary_conditions.top)
m.bind(dpd_left, A, m.Universe.boundary_conditions.left)


uc = m.lattice.sc(a, A)

parts = m.lattice.create_lattice(uc, [25, 25, 25])

print(m.universe.boundary_conditions)

m.show()
