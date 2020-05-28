import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

size = 5

# dimensions of universe
dim=[10., 10., 10.]

# new simulator, don't load any example
m.Simulator(example="", dim=dim)

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff

pot = m.forces.lennard_jones_12_6(0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3)
tstat = m.forces.berenderson_tstat(0.1)

# create a particle type
# all new Particle derived types are automatically
# registered with the universe
class A(m.Particle):
    mass = 39.4
    target_temperature = 270
    
class B(m.Particle):
    mass = 20.4
    target_temperature = 250

m.bind(pot, A, A)
m.bind(pot, A, B)
m.bind(pot, B, B)
m.bind(tstat, Particle)

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(size, 6))
velocities = np.random.normal(0, 5, size=(size,6))

for (pos,vel) in zip(positions, velocities):
    # calling the particle constructor implicitly adds
    # the particle to the universe
    A(pos[:3], vel[:3])
    B(pos[:3], vel[:3])

# run the simulator interactive
m.Simulator.run()
