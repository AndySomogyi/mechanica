import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 3

# number of particles
count = 6000

# number of time points we avg things
avg_pts = 3

# dimensions of universe
dim=np.array([50., 50., 100.])
center = dim / 2

# new simulator
m.Simulator(dim=dim,
            cutoff=cutoff,
            integrator=m.FORWARD_EULER,
            bc=m.BOUNDARY_NONE,
            dt=0.001,
            max_distance=0.2,
            threads=8,
            cells=[5, 5, 5])

clump_radius = 8

class Yolk(m.Particle):
    mass = 500000
    radius = 20
    frozen = True

class Cell(m.Particle):
    mass = 10
    radius = 1.2
    target_temperature=0
    dynamics = m.Overdamped

total_height = 2 * Yolk.radius + 2 * clump_radius
yshift = total_height/2 - Yolk.radius
cshift = total_height/2  - 1.9 * clump_radius

#pot_yc = m.Potential.glj(e=500, r0=10, m=5, k=100, min=0.1, max=50*Cell.radius, tol=0.1)
pot_yc = m.Potential.glj(e=100, r0=5, m=3, k=500, min=0.1, max=50*Cell.radius, tol=0.1)
pot_cc = m.Potential.glj(e=1, r0=2, m=2, min=0.05, max=2.2*Cell.radius)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_yc, Yolk, Cell)
m.Universe.bind(pot_cc, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 100, durration=0.5)

# bind it just like any other force
m.bind(rforce, Cell)

yolk = Yolk(position=center-[0., 0., yshift])

for i, p in enumerate(m.random_points(m.SolidSphere, count)):
    pos = p * clump_radius + center+[0., 0., cshift]
    Cell(position=pos)


# run the simulator interactive
m.Simulator.run()
