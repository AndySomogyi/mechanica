import mechanica as m

m.Simulator(windowless=True)

class Na (m.Particle):
    radius = 0.4
    style={"color":"red"}

class Cl (m.Particle):
    radius = 0.25
    style={"color":"blue"}

uc = m.lattice.bcc(0.9, [Na, Cl])

m.lattice.create_lattice(uc, [10, 10, 10])

# m.system.image_data() is a jpg byte stream of the
# contents of the frame buffer.

with open('system.jpg', 'wb') as f:
    f.write(m.system.image_data())
