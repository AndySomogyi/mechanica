import mechanica as m
import numpy as np

cutoff = 1

m.Simulator(dim=[10., 10., 10.])

class Argon(m.Particle):
    mass = 39.4
    target_temperature = 100

# hook up the destroy method on the Argon type to the
# on_time event
m.on_time(Argon.destroy, period=2, distribution='exponential')

pot = m.Potential.lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, 1.0e-3)
m.Universe.bind(pot, Argon, Argon)

tstat = m.forces.berenderson_tstat(10)

m.Universe.bind(tstat, Argon)

size = 100

# uniform random cube
positions = np.random.uniform(low=0, high=10, size=(size, 3))
velocities = np.random.normal(0, 0.2, size=(size,3))

for (pos,vel) in zip(positions, velocities):
    # calling the particle constructor implicitly adds
    # the particle to the universe
    Argon(pos, vel)

# run the simulator interactive
m.Simulator.run()
