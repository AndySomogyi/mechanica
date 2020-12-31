import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(example="",
            dim=dim,
            cutoff=5,
            integrator=m.FORWARD_EULER,
            dt=0.0005)

class A(m.Particle):
    radius=0.5
    dynamics = m.Overdamped
    mass=5
    style={"color":"MediumSeaGreen"}

class B(m.Particle):
    radius=0.2
    dynamics = m.Overdamped
    mass=10
    style={"color":"skyblue"}

class C(m.Particle):
    radius=10
    frozen = True
    style={"color":"orange"}

pc  = m.Potential.glj(e=10, m=3, max=5)
pa   = m.Potential.glj(e=8, m=3, max=1.0)
pb   = m.Potential.glj(e=1, m=4, max=1)
pab  = m.Potential.harmonic(k=100, r0=0, min=0.01, max=0.55)


# simple harmonic potential to pull particles
h = m.Potential.harmonic(k=300, r0=0.001, max = 5)

m.bind(pc, A, C)
m.bind(pc, B, C)
m.bind(pa, A, A)
#m.bind(pb, B, B)
m.bind(pab, A, B)

r = m.forces.random(0, 5)

m.bind(r, A)
#m.bind(r, B)


c = C(m.Universe.center)

pos_a = m.random_points(m.SolidSphere, 6000, dr=0.45, phi=(0, 0.65 * np.pi)) * ((1 + 0.45/2) * C.radius)  + m.Universe.center

parts, bonds = m.bind_sphere(h, type=B, n=4, phi=(0.6 * np.pi, 0.7 * np.pi), radius = C.radius + B.radius)

[A(p) for p in pos_a]

#m.bind_pairwise(h, c.neighbors(2* A.radius), cutoff=2*A.radius, pairs=[(A,B)])

C.style.visible = True
B.style.visible = True
#A.style.visible = False



c.neighbors(distance=1, types=(A))

def update(e):
    print(B.items().center_of_mass())

m.on_time(update, period=0.01)



m.show()
