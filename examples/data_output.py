import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# new simulator
m.Simulator(dim=[20., 20., 20.])

pot = m.Potential.soft_sphere(kappa=5, epsilon=0.01,
    r0=0.6, eta=3, tol = 0.1, min=0, max=4)

class Cell(m.Particle):
    mass = 20
    target_temperature = 0
    radius=0.5

m.on_time(Cell.fission, period=1, distribution='exponential')

m.Universe.bind(pot, Cell, Cell)

Cell([10., 10., 10.])

# run the simulator interactive
m.Simulator.run()
