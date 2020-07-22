import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# new simulator, don't load any example
m.Simulator(example="", dim=[20., 20., 20.])


pot = m.Potential.soft_sphere(kappa=400, epsilon=5,
    r0=0.7, eta=3, tol = 0.1, min=0.05, max=1.5)

class Cell(m.Particle):
    mass = 20
    target_temperature = 0
    radius=0.5
    events = [m.on_time(m.Particle.fission, period=1, distribution='exponential')]
    dynamics=m.Overdamped

m.Universe.bind(pot, Cell, Cell)

rforce = m.forces.random(0, 0.5)

m.Universe.bind(rforce, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
m.Simulator.run()
