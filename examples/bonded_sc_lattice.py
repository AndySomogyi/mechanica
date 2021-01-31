import mechanica as m
import numpy as np

m.Simulator(dt=0.1)

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

def make_force():
    return [0.5, 1 * np.sin( 0.5 * m.Universe.time), 0]

f = m.forces.ConstantForce(make_force, 0.01)

m.bind(f, B)

pot = m.Potential.power(r0=a/2, alpha=2)

uc = m.lattice.sc(a, A, pot)

parts = m.lattice.create_lattice(uc, [10, 10, 10])

for p in parts[9,:].flatten():
    p[0].become(B)


for p in parts[0,:].flatten():
    p[0].become(Fixed)

m.show()


