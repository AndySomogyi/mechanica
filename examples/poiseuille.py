import mechanica as m
import numpy as np

m.Simulator(dt=0.1, dim=[15, 12, 10], cells=[7, 6, 5], cutoff=0.5,
            bc={'x':'periodic',
                'y':'periodic',
                'top':{'velocity':[0, 0, 0]},
                'bottom':{'velocity':[0, 0, 0]}})

# lattice spacing
a = 0.15

class A (m.Particle):
    radius = 0.05
    style={"color":"seagreen"}
    dynamics = m.Newtonian
    mass=10

dpd = m.Potential.dpd(alpha=10, sigma=1)

m.bind(dpd, A, A)

# driving pressure
pressure = m.forces.ConstantForce([0.1, 0, 0])

m.bind(pressure, A)

uc = m.lattice.sc(a, A)

parts = m.lattice.create_lattice(uc, [40, 40, 40])

m.run()
