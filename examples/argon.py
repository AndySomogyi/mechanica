import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim=[10., 10., 10.]

# new simulator
m.Simulator(dim=dim, window_size=[900,900], perfcounter_period=100)

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = m.Potential.lennard_jones_12_6(0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3 )


# create a particle type
# all new Particle derived types are automatically
# registered with the universe
class Argon(m.Particle):
    radius=0.1
    mass = 39.4

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot, Argon, Argon)

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(10000, 3))

for pos in positions:
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos)

# run the simulator interactive
m.Simulator.run()








