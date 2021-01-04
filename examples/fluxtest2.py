import mechanica as m

m.Simulator(dim=[6.5, 6.5, 6.5], bc=m.FREESLIP_FULL)

class A (m.Particle):
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1",
                             "map" : "rainbow",
                             "range" : "auto"}}
    radius = 0.1

uc = m.lattice.sc(0.25, A)

parts = m.lattice.create_lattice(uc, [25, 25, 25])

m.flux(A, A, "S1", 5)

parts[24,24,24][0].species.S1 = 5000

m.show()
