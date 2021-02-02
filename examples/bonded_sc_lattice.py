import mechanica as m
import numpy as np

m.Simulator(dt=0.1, dim=[15, 12, 10]) #, bc=m.FREESLIP_FULL)

# lattice spacing
a = 0.65

class A (m.Particle):
    radius = 0.3
    style={"color":"seagreen"}
    dynamics = m.Overdamped

class B (m.Particle):
    radius = 0.3
    style={"color":"red"}
    dynamics = m.Overdamped

class Fixed (m.Particle):
    radius = 0.3
    style={"color":"blue"}
    frozen = True

repulse = m.Potential.coulomb(q=0.08, min=0.01, max=2*a)

m.bind(repulse, A, A)
m.bind(repulse, A, B)

f = m.forces.ConstantForce(
    lambda: [0.3, 1 * np.sin( 0.4 * m.Universe.time), 0], 0.01)

m.bind(f, B)

pot = m.Potential.power(r0=0.5*a, alpha=2)

uc = m.lattice.sc(a, A,
                  lambda i, j: m.Bond(pot, i, j, dissociation_energy=1.3))

parts = m.lattice.create_lattice(uc, [15, 15, 15])

for p in parts[14,:].flatten(): p[0].become(B)

for p in parts[0,:].flatten(): p[0].become(Fixed)

m.show()


