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
import _mechanica



print ("mechanica file: " + m.__file__, flush=True)
print("_mechanica file: " + _mechanica.__file__, flush=True)

c = m.Simulator.Config()
s = m.Simulator()

pot = m.Potential.coulomb(0.01, 10, 1)

print("getting particle")
x = m.Universe.particles[0]

print("n part: ", len(m.Universe.particles))

print("creating new base particle")
p = m.Particle([1.,2.,3.])

print("n part: ", len(m.Universe.particles))

print("new part position")
print(p.position)

print("creating new type")
class B(m.Particle):
    pass

m.Universe.bind(pot, B, B)

print("creating new derived type")
b = B()

print(b)

print("getting particle")
p = m.Universe.particles[0]

print("printing the particle")

print(p)

ar = type(p)

print("getting instance charge")
print(p.charge)

print("setting type.charge")
ar.charge = 9

print("getting instance charge")
print(p.charge)

print("setting instance charge")
p.charge = 11

print("getting type charge")
print(ar.charge)

print("setting p to none")
p = None

print("all done, calling exit")

exit()









