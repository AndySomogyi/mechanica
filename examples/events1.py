import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim=[10., 10., 10.]

# new simulator
m.Simulator(dim=dim)

class MyCell(m.Particle):

    mass = 39.4
    target_temperature = 50
    radius = 0.2

    def __init__(self, *args):
        super().__init__(*args)
        print("creating new particle, my id is: ", self.id)

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff
pot = m.Potential.lennard_jones_12_6(0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3 )


# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot, MyCell, MyCell)

# create a thermostat, coupling time constant determines how rapidly the
# thermostat operates, smaller numbers mean thermostat acts more rapidly
tstat = m.forces.berenderson_tstat(10)

# bind it just like any other force
m.Universe.bind(tstat, MyCell)

# create a new particle every 0.05 time units. The 'on_time' event
# here binds the constructor of the MyCell object with the event, and
# calls at periodic intervals based on the exponential distribution,
# so the mean time between particle creation is 0.05
m.on_time(MyCell, period=0.05, distribution="exponential")

# run the simulator interactive
m.Simulator.run()
