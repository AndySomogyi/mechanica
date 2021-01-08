import mechanica as m
import numpy as n

m.Simulator()



class A(m.Particle):

    radius = 1

    species = ['S1', 'S2', 'S3']

    style = {"colormap" : {"species" : "S2", "map" : "rainbow", "range" : "auto"}}


class B(m.Particle):

    radius = 4

    species = ['S2', 'S3', 'S4']

    style = {"colormap" : {"species" : "S2", "map" : "rainbow", "range" : "auto"}}


o = A()

o.species.S2 = 0.5

o.become(B)

m.show()
