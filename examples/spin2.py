import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(dim=dim,
            cutoff=10,
            integrator=m.FORWARD_EULER,
            cells=[3, 3, 3],
            dt=0.001)

class A(m.Particle):
    radius=1
    dynamics = m.Newtonian
    mass=20
    style={"color":"MediumSeaGreen"}


p = m.Potential.glj(e=0.1, m=3, max=3)
cp = m.Potential.coulomb(q=5000, min=0.05, max=10)

m.bind(p, A, m.Cuboid)
m.bind(cp, A, A)

rforce = m.forces.friction(0.01, 0, 100)

# bind it just like any other force
m.bind(rforce, A)

c = m.Cuboid(m.Universe.center + [0, 0, 0], size=[25, 31, 5])

c.spin = [0.0, 8.5, 0.0]

# uniform random cube
positions = np.random.uniform(low=0, high=30, size=(2500, 3))

for p in positions:
    A(p, velocity = [0, 0, 0])

m.show()
