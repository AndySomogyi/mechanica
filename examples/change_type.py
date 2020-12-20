import mechanica as m
import numpy as np

m.Simulator()

class A(m.Particle):
    radius=0.1
    dynamics = m.Overdamped
    mass=5
    style={"color":"MediumSeaGreen"}

class B(m.Particle):
    radius=0.1
    dynamics = m.Overdamped
    mass=10
    style={"color":"skyblue"}

p = m.Potential.coulomb(q=2, min=0.01, max=3)

m.bind(p, A, A)
m.bind(p, B, B)
m.bind(p, A, B)

r = m.forces.random(0, 1)

m.bind(r, A)

pos = m.random_points(m.SolidCube, 50000) * 10 + m.Universe.center

[A(p) for p in pos]

a = A.items()[0]

[p.become(B) for p in a.neighbors(3)]

a.radius = 2

m.show()
