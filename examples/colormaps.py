import mechanica as m

m.Simulator()

class Bead(m.Particle):
    species = ['S1']
    radius = 3

    style = {"colormap" : {"species" : "S1", "map" : "rainbow", "range" : "auto"}}

    def __init__(self, pos, value):
        super().__init__(pos)
        self.species.S1 = value

# make a ring of of 50 particles
pts = m.points(m.Ring, 100) * 4 + m.Universe.center

# constuct a particle for each position, make
# a list of particles
beads = [Bead(p, i/100.) for i, p in enumerate(pts)]

Bead.i = 0

def keypress(e):
    names = m.Colormap.names()
    name = None

    if (e.key_name == "n"):
        Bead.i = (Bead.i + 1) % len(names)
        name = names[Bead.i]
        print("setting colormap to: ", name)

    elif (e.key_name == "p"):
        Bead.i = (Bead.i - 1) % len(names)
        name = names[Bead.i]
        print("setting colormap to: ", name)

    Bead.style.colormap = name


m.on_keypress(keypress)

# run the model
m.show()
