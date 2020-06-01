import mechanica as m
import numpy as np

# create one type of particle, 
class A(m.Particle):
    species = ['S1', 'S2', 'S3']
    mass = 1
    
# another type, with some reactions
class B(m.Particle):
    species = {'S1' :Species(boundary=True, 1.0), 
               'Foo':Species(5.0), 
               'Bar':Species(10.0)}
    reactions = ['S1 + S1 -> S3; k*S1*S2', 'S3->0; k2*S3']
    
# create a flux between species at particles
f = m.fluxes.fick(k=0.5, omega='gaussian')
m.Universe.bind(f, A.S1, B.Foo)
m.Universe.bind(f, A.S3, B.Foo)

for p in m.Universe.particles: 
    print(p.S1)

pos = np.random.uniform(low=0, high=10, size=(4, 3))
vel = np.random.normal(-5, 5, size=(4,3))

class P(m.Particle):
    mass = 39
    
p1 = P(x=pos[0], v=vel[0])
p2 = P(x=pos[1], v=vel[1])
p3 = P(x=pos[2], v=vel[2])
p4 = P(x=pos[3], v=vel[3])

b = m.forces.harmonic_bond(k=-0.1)
a = m.forces.angle(k=0.3)
d = m.forces.dihedral(k=0.23)

m.Universe.bind(b, p1, p2)
m.Universe.bind(a, p1, p2, p3)
m.universe.bind(d, p1, p2, p3, p4)


c = m.collisions.collision(deltaG = 23.5, e = 15)
m.Universe.bind(c, Li, F, LiF)
