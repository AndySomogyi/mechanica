import mechanica as m
import numpy as np

m.Simulator(dt=0.1, dim=[15, 12, 10],
            bc={'x':'periodic', 'y':'freeslip', 'z':'freeslip'},
            perfcounter_period=100)

# lattice spacing
a = 0.7

class A (m.Particle):
    radius = 0.3
    style={"color":"seagreen"}
    dynamics = m.Newtonian
    mass=10

dpd = m.Potential.dpd(sigma=1.5)

m.bind(dpd, A, A)

f = m.forces.ConstantForce([0.005, 0, 0])

m.bind(f, A)

uc = m.lattice.sc(a, A)

parts = m.lattice.create_lattice(uc, [15, 15, 15])

m.run()
