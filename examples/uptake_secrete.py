import mechanica as m

m.init(dim=[6.5, 6.5, 6.5], bc=m.FREESLIP_FULL)

class A (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class Producer (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class Consumer (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

# define fluxes between objects types
m.flux(A, A, "S1", 5, 0)
m.produce_flux(Producer, A, "S1", 5, 0)
m.consume_flux(A, Consumer, "S1", 10, 500)

# make a lattice of objects
uc = m.lattice.sc(0.25, A)
parts = m.lattice.create_lattice(uc, [25, 25, 1])

# grap the left part
left = parts[0,12,0][0]

# grab the right part
right = parts[24,12,0][0]

# change types
left.become(Producer)
right.become(Consumer)

left.species.S1 = 200 # set initial condition

m.show()
