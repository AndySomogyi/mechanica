import mechanica as m

m.Simulator()

class Na (m.Particle):
    radius = 0.4
    style={"color":"green"}

class Cl (m.Particle):
    radius = 0.25
    style={"color":"purple"}

uc = m.lattice.bcc(0.9, [Na, Cl])

m.lattice.create_lattice(uc, [10, 10, 10])

m.show()
