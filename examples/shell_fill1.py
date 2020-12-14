import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 10

# number of particles
count = 50000

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


class Big(m.Particle):
    mass = 500000
    radius = 20
    frozen = True

class Small(m.Particle):
    mass = 10
    radius = 0.25
    target_temperature=0
    dynamics = m.Overdamped

pot_yc = m.Potential.glj(e=100, r0=5, m=3, k=500, min=0.1,  max=1.5 * Big.radius, tol=0.1)
#pot_cc = m.Potential.glj(e=0,   r0=2, m=2, k=10,  min=0.05, max=1 * Big.radius)
pot_cc = m.Potential.harmonic(r0=0, k=0.1, max=10)

#pot_yc = m.Potential.glj(e=10, r0=1, m=3, min=0.1, max=50*Small.radius, tol=0.1)
#pot_cc = m.Potential.glj(e=100, r0=5, m=2, min=0.05, max=0.5*Big.radius)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_yc, Big, Small)
m.Universe.bind(pot_cc, Small, Small)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 100, durration=0.5)

# bind it just like any other force
m.bind(rforce, Small)

yolk = Big(position=center)


for p in m.random_points(m.Sphere, count):
    pos = p * (Big.radius + Small.radius) + center
    Small(position=pos)


# run the simulator interactive
m.Simulator.run()
