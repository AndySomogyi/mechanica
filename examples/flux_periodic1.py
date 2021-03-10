import mechanica as m

m.init(dt=0.1, dim=[15, 5, 5], cutoff = 3,
       bc={'x':('periodic','reset')})

class A(m.Particle):
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1",
                           "map" : "rainbow",
                           "range" : "auto"}}

m.flux(A, A, "S1", 2)

a1 = A(m.universe.center - [0, 1, 0])
a2 = A(m.universe.center + [-5, 1, 0], velocity=[0.5, 0, 0])

a1.species.S1 = 3
a2.species.S1 = 0

m.show()
