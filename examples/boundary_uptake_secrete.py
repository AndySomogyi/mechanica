import mechanica as m

m.init(dim=[6.5, 6.5, 6.5], bc=m.FREESLIP_FULL)

class A (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class Producer (m.Particle):
    radius = 0.1
    species = ['S1=200', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class Consumer (m.Particle):
    radius = 0.1
    species = ['S1', 'S2', 'S3']
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class Source (m.Particle):
    radius = 0.1
    species = ['$S1=5', 'S2', 'S3'] # make S1 a boundary species here.
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

class Sink (m.Particle):
    radius = 0.1
    species = ['$S1', 'S2', 'S3'] # make S1 a boundary species here.
    style = {"colormap" : {"species" : "S1", "map" : "rainbow","range" : "auto"}}

# define fluxes between objects types
m.flux         (A, A, "S1", 5, 0.001)
m.secrete_flux (Producer, A, "S1", 5, 0)
m.uptake_flux  (A, Consumer, "S1", 10, 500)
m.secrete_flux (Source, A, "S1", 5, 0)
m.flux         (Sink, A, "S1", 200, 0)

# make a lattice of objects
uc = m.lattice.sc(0.25, A)
parts = m.lattice.create_lattice(uc, [25, 25, 1])

# grap the left part, make it a producer
parts[0,12,0][0].become(Producer)

# grab the right part, make it a consumer
parts[24,12,0][0].become(Consumer)

# grab the middle, make it a source
parts[12, 12, 0][0].become(Source)

# make the corner parts sinks
parts[0,  0, 0][0].become(Sink)
parts[0, 24, 0][0].become(Sink)
parts[24, 0, 0][0].become(Sink)
parts[24,24, 0][0].become(Sink)

# set the species values to thier init conditions based on type definitions.
m.reset_species()

# make an event handler to listen for key press
def key(e):
    if e.key_name == "x":
        m.reset_species()

m.on_keypress(key)

m.show()
