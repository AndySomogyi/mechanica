import mechanica as m
import numpy as n

m.Simulator()

print(m.carbon.Species("S1"))

s1 = m.carbon.Species("$S1")

s1 = m.carbon.Species("const S1")

s1 = m.carbon.Species("const $S1")

s1 = m.carbon.Species("S1 = 1")

s1 = m.carbon.Species("const S1 = 234234.5")


class A(m.Particle):

    species = ['S1', 'S2', 'S3']

    style = {"colormap" : {"species" : "S1", "map" : "rainbow", "range" : "auto"}}


m.flux(A, A, "S1", 5)

print("A.species:")
print(A.species)

print("making f")
a1 = A(m.Universe.center)

a2 = A(m.Universe.center + [0, 0.5, 0])

a1.species.S1 = 0

a2.species.S1 = 1


print("A.species.S1: ", A.species.S1)

m.show()
