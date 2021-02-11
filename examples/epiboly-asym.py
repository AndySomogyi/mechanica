import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(dim=dim,
            cutoff=5,
            integrator=m.FORWARD_EULER,
            dt=0.001)

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
pa   = m.Potential.glj(e=2, m=2, max=3.0)
pb   = m.Potential.glj(e=1, m=4, max=1)
pab  = m.Potential.harmonic(k=10, r0=0, min=0.01, max=0.55)


# simple harmonic potential to pull particles
h = m.Potential.harmonic(k=40, r0=0.001, max = 5)

m.bind(pc, A, C)
m.bind(pc, B, C)
m.bind(pa, A, A)
#m.bind(pb, B, B)
#m.bind(pab, A, B)

r = m.forces.random(0, 5)

m.bind(r, A)
#m.bind(r, B)


c = C(m.Universe.center)

pos_a = m.random_points(m.SolidSphere, 3000, dr=0.25, phi=(0, 0.60 * np.pi)) \
    * ((1 + 0.25/2) * C.radius)  + m.Universe.center

parts, bonds = m.bind_sphere(h, type=B, n=4, phi=(0.6 * np.pi, np.pi), radius = C.radius + B.radius)

[A(p) for p in pos_a]

# grab a vertical slice of the neighbors of the yolk:
slice_parts = [p for p in c.neighbors() if p.spherical()[1] > 0]

m.bind_pairwise(pab, slice_parts, cutoff=5*A.radius, pairs=[(A,B)])

#A.style.visible = False

C.style.visible = False
B.style.visible = False
#A.style.visible = False




def update(e):
    print(B.items().center_of_mass())

m.on_time(update, period=0.01)



m.show()
