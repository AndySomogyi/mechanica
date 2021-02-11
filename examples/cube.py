import mechanica as m
import numpy as np



# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(dim=dim,
            cutoff=10,
            integrator=m.FORWARD_EULER,
            cells=[1, 1, 1],
            dt=0.01)

class A(m.Particle):
    radius=0.5
    dynamics = m.Newtonian
    mass=5
    style={"color":"MediumSeaGreen"}

class Sphere(m.Particle):
    radius=3
    frozen = True
    style={"color":"orange"}

class Test(m.Particle):
    radius=0
    frozen = True
    style={"color":"orange"}


p = m.Potential.glj(e=30, m=2, max=10)

m.bind(p, A, Sphere)
m.bind(p, A, Test)
m.bind(p, A, m.Cuboid)


# above the sphere
Sphere(m.Universe.center + [5, 0, 0])
A(m.Universe.center + [5, 0, 5.8])

# above the test
Test(m.Universe.center + [0, -10, 3])
A(m.Universe.center + [0, -10, 5.8])

# above the scube
c = m.Cuboid(m.Universe.center + [-5, 0, 0], size=[6, 6, 6])
A(m.Universe.center + [-5, 0, 5.8])


m.show()
