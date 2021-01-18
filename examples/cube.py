import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim=[10., 10., 10.]

# new simulator, don't load any example
m.Simulator(dim=dim, window_size=[900,900])

c = m.Cuboid(pos=m.Universe.center + [1, 1, 1])

# run the simulator interactive
m.Simulator.show()
