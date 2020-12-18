import mechanica as m

m.Simulator()

print(m.carbon.Species("S1"))

s1 = m.carbon.Species("$S1")

s1 = m.carbon.Species("const S1")

s1 = m.carbon.Species("const $S1")

s1 = m.carbon.Species("S1 = 1")

s1 = m.carbon.Species("const S1 = 234234.5")


class A(m.Particle):
    species = ['S1', 'S2', 'S3']

print("A.species:")
print(A.species)

print("making f")
a = A()

print("f.species")
print(a.species)
