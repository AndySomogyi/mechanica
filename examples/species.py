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

    def update(self, time):
        self.species.S1 = (1 + n.sin(2. * time))/2

m.on_time(A.update, period=0.01)


print("A.species:")
print(A.species)

print("making f")
a = A()

print("f.species")
print(a.species)


print("A.species.S1: ", A.species.S1)

m.show()
