import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# new simulator, don't load any example
m.Simulator(example="", dim=[20., 20., 20.])

pot = m.Potential.soft_sphere(kappa=500, epsilon=1000,
    r0=0.6, eta=3, tol = 0.1, min=0.05, max=4)

class Cell(m.Particle):
    mass = 20
    target_temperature = 0
    radius=0.5
    events = [m.on_time(m.Particle.fission, period=1, distribution='exponential')]

m.Universe.bind(pot, Cell, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
m.Simulator.run()
