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

c.windowless = False

s = m.Simulator(c, foo='bar', bar=1)

print ("renderer: " , Simulator.renderer)

print("s.foo: ", s.foo)





foo = _mechanica.Foo("foo")



foo.stuff(this="that", stuff="this")




class S(ctypes.Structure) : pass
class P(m.Particle) : pass

print("getting P mass", flush=True)
print(P.mass)

print("getting m.Particle.mass", flush=True)
print(m.Particle.mass)

print("creating P instance...", flush=True)

p = P()

print("setting p mass", flush=True)
P.mass = 5

print("getting P mass", flush=True)
print(P.mass)

print("creating p", flush=True)
p = P()

print("getting p mass")
print(p.mass)


def bumpVec(p):
    p.position[0] += 1
    print(p.position)
    
    
for i in range(5):
    bumpVec(p)
    
    
print("final foo:")
print(p.position)




