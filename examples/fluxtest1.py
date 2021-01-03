import mechanica as m
import numpy as n

m.Simulator()

class A(m.Particle):
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1",
                           "map" : "rainbow",
                           "range" : "auto"}}

m.flux(A, A, "S1", 5)

a1 = A(m.Universe.center)
a2 = A(m.Universe.center + [0, 0.5, 0])

a1.species.S1 = 0
a2.species.S1 = 1

m.show()
