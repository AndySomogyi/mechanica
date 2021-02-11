import mechanica as m
import numpy as np



# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(dim=dim,
            cutoff=10,
            integrator=m.FORWARD_EULER,
            cells=[1, 1, 1],
            dt=0.005)

class A(m.Particle):
    radius=0.5
    dynamics = m.Newtonian
    mass=30
    style={"color":"MediumSeaGreen"}

class Sphere(m.Particle):
    radius=3
    frozen = True
    style={"color":"orange"}

class Test(m.Particle):
    radius=0
    frozen = True
    style={"color":"orange"}


p = m.Potential.glj(e=1, m=2, max=10)

m.bind(p, A, Sphere)
m.bind(p, A, Test)
m.bind(p, A, m.Cuboid)
m.bind(p, A, A)


# above the sphere
#Sphere(m.Universe.center + [5, 0, 0])

#A(m.Universe.center + [5, 0, 5.8])

# above the test
Test(m.Universe.center + [0, -10, 3])
#A(m.Universe.center + [0, -10, 5.8])

# above the scube
c = m.Cuboid(m.Universe.center + [0, 0, 0],
             size=[13, 13, 15],
             orientation=[0, -np.pi/1.8, 0])

c.rotate([0, 0.1, 0])

c.spin = [0, 0.2, 0]


m.show()
