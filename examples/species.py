import mechanica as m

m.Simulator()

class A(m.Particle):
    species = ['S1', 'S2', 'S3']

print("A.species:")
print(A.species)

print("making f")
a = A()

print("f.species")
print(a.species)
