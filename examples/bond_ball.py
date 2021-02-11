import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])

m.Simulator(dim=dim,
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
    mass=1
    style={"color":"skyblue"}

class C(m.Particle):
    radius=10
    frozen = True
    style={"color":"orange"}

C(m.Universe.center)

pos_a = m.random_points(m.Sphere, 12000) * (C.radius+A.radius)  + m.Universe.center
pos_b = m.random_points(m.Sphere, 4000) * (C.radius+B.radius + 1)  + m.Universe.center

# make a ring of of 50 particles
pts = m.points(m.Ring, 100) * (C.radius+B.radius)  + m.Universe.center - [0, 0, 1]

#positions = m.random_points(m.SolidSphere, 5000) * 9 + m.Universe.center

#[A(p) for p in pos_a if p[2] > m.Universe.center[2]]
[B(p) for p in pts]


pc  = m.Potential.glj(e=30, m=2, max=5)
pa   = m.Potential.glj(e=3, m=2.5, max=3)
pb   = m.Potential.glj(e=1, m=4, max=1)
pab  = m.Potential.glj(e=1, m=2, max=1)
ph = m.Potential.harmonic(r0=0.001, k=200)

m.bind(pc, A, C)
m.bind(pc, B, C)
m.bind(pa, A, A)
#m.bind(pb, B, B)
m.bind(pab, A, B)

r = m.forces.random(0, 5)

m.bind(r, A)
m.bind(r, B)

#m.bind_pairwise(ph, [p for p in m.Universe.particles if p.position[2] < m.Universe.center[2] + 1], 2)
m.bind_pairwise(ph, B.items(), 1)
#m.bind_pairwise(ph, B.items(), 2)

def update(e):
    
    print(B.items().center_of_mass())

m.on_time(update, period=0.01)



m.show()
