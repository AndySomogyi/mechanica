import mechanica as m
import numpy as np

m.Simulator(dt=0.1, bc=m.FREESLIP_FULL)

# lattice spacing
a = 1.0

class A (m.Particle):
    radius = 0.4
    style={"color":"green"}
    dynamics = m.Overdamped

class B (m.Particle):
    radius = 0.4
    style={"color":"red"}
    dynamics = m.Overdamped

class Fixed (m.Particle):
    radius = 0.4
    style={"color":"blue"}
    frozen = True


repulse = m.Potential.coulomb(q=1, min=0.05, max=2)

m.bind(repulse, A, A)
m.bind(repulse, A, B)
m.bind(repulse, A, Fixed)

def make_force():
    return [0.48, 1 * np.sin( 0.4 * m.Universe.time), 0]

f = m.forces.ConstantForce(make_force, 0.01)

m.bind(f, B)

pot = m.Potential.power(r0=a/2, alpha=2)

uc = m.lattice.sc(a, A, lambda i, j: m.Bond(pot, i, j, dissociation_energy=1.21))

parts = m.lattice.create_lattice(uc, [10, 10, 10])

for p in parts[9,:].flatten():
    p[0].become(B)


for p in parts[0,:].flatten():
    p[0].become(Fixed)

m.show()


