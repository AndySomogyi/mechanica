import mechanica as m
import numpy as np

# dimensions of universe
dim=[20., 20., 20.]

# new simulator
m.Simulator(dim=dim)

# loop over arays for x/y coords
for x in np.arange(0., 20., 2.):
    for y in np.arange(0., 20., 2.):

        # create a new particle type, chooses next default color
        class P(m.Particle):
            radius = 2

        # instantiate that type
        P([x+1.5, y+1.5, 10.])

# run the simulator interactive
m.Simulator.irun()
