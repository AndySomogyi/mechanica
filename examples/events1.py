import mechanica as m
import numpy as np

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim=[10., 10., 10.]

# new simulator, don't load any example
m.Simulator(example="", dim=dim)


# create a particle type, all new Particle derived types
# are automatically registered with the universe

        
class Receptor(m.Particle):
    mass = 0.1
    
    
class Virus(m.Particle):
    mass = 0.2
    
    
class MyCell(m.Particle):

    mass = 39.4
    target_temperature = 50

    def __init__(self, *args):
        super().__init__(*args)
        print("creating new particle, my id is: ", self.id)
        
# create a new cell instance
cell = MyCell(center)
    
harmonic = m.Harmonic(k=300, r0=MyCell.radius)
    
# create new receptors on the surface of the cell,
# bind them to the cell with explicit harmonic potential
for p in m.random_points(m.Sphere, 300) * MyCell.radius + center:
    r = Receptor(p)
    m.bind(harmonic, cell, r)
    

ss = Potential.soft_sphere(kappa=10, epsilon=50, r0=3, eta=3, tol = 0.1, min=0.1, max=9)

# create a bond creation event
r = ReactivePotential(potential=ss, reaction=Bond)

# can bind any operator to a reactive potential
r = ReactivePotential(potential=ss, reaction=Fission)


# stable explicit bond, can use arbitrary potential function
b = Bond(potential=Harmonic(k=50, r0=2.0))

# optional half-life
b = Bond(potential=Harmonic(k=50, r0=2.0) 
         half-life = 30)

# optional bond-strength breaking strength
b = Bond(potential=Harmonic(k=50, r0=2.0) 
         bond-strength = 15)
    
    
    
    



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
