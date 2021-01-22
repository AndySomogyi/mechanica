import mechanica as m
import numpy as np


# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(example="",
            dim=dim,
            cutoff=10,
            integrator=m.FORWARD_EULER,
            cells=[1, 1, 1],
            dt=0.001)

class A(m.Particle):
    radius=1
    dynamics = m.Overdamped
    mass=20
    style={"color":"MediumSeaGreen"}

class Sphere(m.Particle):
    radius=3
    frozen = True
    style={"color":"orange"}

class Test(m.Particle):
    radius=0
    frozen = True
    style={"color":"orange"}


p = m.Potential.glj(e=0.1, m=3, max=3)

m.bind(p, A, Sphere)
m.bind(p, A, Test)
m.bind(p, A, m.Cuboid)
m.bind(p, A, A)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 200)

# bind it just like any other force
m.bind(rforce, A)


# above the sphere
#Sphere(m.Universe.center + [5, 0, 0])

#A(m.Universe.center + [5, 0, 5.8])

# above the test
Test(m.Universe.center + [0, -10, 3])
#A(m.Universe.center + [0, -10, 5.8])

# above the scube
c = m.Cuboid(m.Universe.center + [0, 0, 0],
             size=[25, 31, 5],
             orientation=[0, -np.pi/1.8, 0])

c.rotate([0, 0.1, 0])

c.spin = [0.0, 1.2, 0.0]

# uniform random cube
positions = np.random.uniform(low=0, high=30, size=(2000, 3))

for p in positions:
    A(p, velocity = [0, 0, 0])




m.show()
