import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 8

count = 3000

# dimensions of universe
dim=[20., 20., 20.]
center = np.array([10., 10., 10.])

# new simulator, don't load any example
m.Simulator(example="", dim=dim, cutoff=cutoff)

class Big(m.Particle):
    mass = 500000
    radius = 3

class Small(m.Particle):
    mass = 0.1
    radius = 0.2
    target_temperature=0
    dynamics = m.Overdamped

pot_bs = m.Potential.soft_sphere(kappa=10, epsilon=50, r0=2.9, eta=3, tol = 0.1, min=0.1, max=9)
pot_ss = m.Potential.soft_sphere(kappa=20, epsilon=0.0001, r0=0.2, eta=2, tol = 0.05, min=0.01, max=3)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_bs, Big, Small)
m.Universe.bind(pot_ss, Small, Small)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = m.forces.random(0, 0.00000001)

# bind it just like any other force
#m.Universe.bind(tstat, Small)

#center = np.array([5., 5., 5.])

Big(position=center, velocity=[0., 0., 0.])

thickness = 1.

blob_center = center + [0, Big.radius, 0]

while(len(m.Universe.particles) < count) :
    pos = np.random.uniform(low=center[0]-4, high=center[0]+4, size=(3))
    pos[1] = center[0]+3
    Small(pos)
    #dist = np.linalg.norm(pos - blob_center)
    #if(dist < 2):
    #    Small(pos)


# run the simulator interactive
m.Simulator.run()








