import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])
center = dim / 2

m.Simulator(dim=dim,
            cutoff=10,
            integrator=m.FORWARD_EULER,
            dt=0.0005)

class C(m.Cluster):
    radius=3

    class A(m.Particle):
        radius=0.5
        dynamics = m.Overdamped
        mass=10
        style={"color":"MediumSeaGreen"}

    class B(m.Particle):
        radius=0.5
        dynamics = m.Overdamped
        mass=10
        style={"color":"skyblue"}


c1 = C(position=center - (3, 0, 0))
c2 = C(position=center + (7, 0, 0))

c1.A(2000)
c2.B(2000)

p1  = m.Potential.glj(e=7, m=1, max=1)
p2  = m.Potential.glj(e=7, m=1, max=2)
m.bind(p1, C.A, C.A, bound=True)
m.bind(p2, C.B, C.B, bound=True)

rforce = m.forces.random(0, 10)
m.bind(rforce, C.A)
m.bind(rforce, C.B)

m.run()
