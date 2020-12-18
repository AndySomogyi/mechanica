import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator()

class A(m.Particle):
    radius=0.2
    dynamics = m.Overdamped
    mass=5
    style={"color":"MediumSeaGreen"}

class B(m.Particle):
    radius=0.2
    dynamics = m.Overdamped
    mass=10
    style={"color":"skyblue"}

p   = m.Potential.glj(e=8, m=3, max=1.0)

m.bind(p, A, A)
m.bind(p, B, B)
m.bind(p, A, B)

r = m.forces.random(0, 5)

m.bind(r, A)

pos_a = m.random_points(m.SolidSphere, 6000, dr=0.45, phi=(0, 0.65 * np.pi)) * ((1 + 0.45/2) * C.radius)  + m.Universe.center

parts, bonds = m.bind_sphere(h, type=B, n=4, phi=(0.6 * np.pi, 0.7 * np.pi), radius = C.radius + B.radius)

[A(p) for p in pos_a]

#m.bind_pairwise(h, c.neighbors(2* A.radius), cutoff=2*A.radius, pairs=[(A,B)])

C.style.visible = False
B.style.visible = False
#A.style.visible = False



c.neighbors(distance=1, types=(A))

def update(e):
    print(B.items().center_of_mass())

m.on_time(update, period=0.01)



m.show()
