import mechanica as m
import numpy as n

m.Simulator()

class A(m.Particle):

    radius = 5

    species = ['S1', 'S2', 'S3']

    style = {"colormap" : {"species" : "S1", "map" : "rainbow", "range" : "auto"}}

    def update(self, time):
        self.species.S1 = (1 + n.sin(2. * time))/2

m.on_time(A.update, period=0.01)

a = A(m.Universe.center)

m.run()
