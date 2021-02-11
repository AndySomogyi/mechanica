import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(dim=dim,
            cutoff=5,
            integrator=m.FORWARD_EULER,
            dt=0.001)

class A(m.Particle):
    radius=0.1
    dynamics = m.Overdamped
    mass=5
    style={"color":"MediumSeaGreen"}

class C(m.Particle):
    radius=10
    frozen = True
    style={"color":"orange"}

C(m.Universe.center)

pos = m.random_points(m.Sphere, 5000) * (C.radius+A.radius)  + m.Universe.center

[A(p) for p in pos]

pc = m.Potential.glj(e=30, m=2, max=5)
pa = m.Potential.coulomb(q=100, min=0.01, max=5)

m.bind(pc, A, C)
m.bind(pa, A, A)

m.show()
