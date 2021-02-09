import mechanica as m
import numpy as np

m.Simulator(dt=0.1, dim=[15, 12, 10],
            bc=m.PERIODIC_X | m.FREESLIP_Y | m.FREESLIP_Z) #, bc=m.FREESLIP_FULL)

# lattice spacing
a = 0.9

class A (m.Particle):
    radius = 0.3
    style={"color":"seagreen"}
    dynamics = m.Newtonian
    mass=10

dpd = m.Potential.dpd(sigma=0.1)

m.bind(dpd, A, A)

f = m.forces.ConstantForce([0.01, 0, 0])

m.bind(f, A)

uc = m.lattice.sc(a, A)

parts = m.lattice.create_lattice(uc, [10, 10, 10])


m.run()
