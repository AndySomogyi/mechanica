import ctypes
import sys
import os.path as path


cur_dir = path.dirname(path.realpath(__file__))

print(cur_dir)

print(sys.path)

sys.path.append("/Users/andy/src/mechanica/src")
sys.path.append("/Users/andy/src/mx-xcode/Debug/lib")
#sys.path.append("/Users/andy/src/mx-eclipse/src")

print(sys.path)



import mechanica as m
import numpy as np
import _mechanica

# potential cutoff distance
cutoff = 1

# dimensions of universe
dim=[10., 10., 10.]

    

print ("mechanica file: " + m.__file__, flush=True)
print("_mechanica file: " + _mechanica.__file__, flush=True)


# new simulator, don't load any example 
m.Simulator(example="", dim=dim)

# create a potential representing a 12-6 Lennard-Jones potential
# A The first parameter of the Lennard-Jones potential.
# B The second parameter of the Lennard-Jones potential.
# cutoff 
pot = m.Potential.lennard_jones_12_6(0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3 ) 


# create a particle type
# all new Particle derived types are automatically
# registered with the universe
class Argon(m.Particle):
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








