import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim=[10., 10., 10.]

# new simulator
m.Simulator(dim=dim)

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = m.Potential.lennard_jones_12_6(0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3 )

# create a particle type
# all new Particle derived types are automatically
# registered with the universe
class Argon(m.Particle):
    mass = 39.4
    target_temperature = 10000

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot, Argon, Argon)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = m.forces.berenderson_tstat(10)

# bind it just like any other force
m.Universe.bind(tstat, Argon)

size = 100

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(size, 3))
velocities = np.random.normal(0, 0.1, size=(size,3))

for (pos,vel) in zip(positions, velocities):
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos, vel)

# run the simulator interactive
m.Simulator.run()
