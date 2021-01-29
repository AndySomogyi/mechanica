import mechanica as m

m.Simulator(dim=[6.5, 6.5, 6.5], bc=m.FREESLIP_FULL)

class A (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class B (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

    def spew(self, event):

        print("spew...")

        # reset the value of the species
        # secrete consumes material...
        self.species.S1 = 500
        self.species.S1.secrete(250, distance=1)

m.flux(A, A, "S1", 5, 0.005)

uc = m.lattice.sc(0.25, A)

parts = m.lattice.create_lattice(uc, [25, 25, 25])

# grap the particle at the top cornder
o = parts[24,0,24][0]

print("secreting pos: ", o.position)

# change type to B, since there is no flux rule between A and B
o.become(B)

m.on_time(B.spew, period=0.3)
m.show()
