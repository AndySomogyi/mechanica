import mechanica as m

m.Simulator()

class Foo(m.Particle):
    species = ['S1', 'S2', 'S3']


print("Foo.species:")
print(Foo.species)

print("making f")
f = Foo()

print("f.species")
print(f.species)

m.show()
