import mechanica as m

m.Simulator(windowless=True, window_size=[2048,2048])

class Na (m.Particle):
    radius = 0.4
    style={"color":"orange"}

class Cl (m.Particle):
    radius = 0.25
    style={"color":"dodgerblue"}

uc = m.lattice.bcc(0.9, [Na, Cl])

m.lattice.create_lattice(uc, [10, 10, 10])

# m.system.image_data() is a jpg byte stream of the
# contents of the frame buffer.

with open('system.jpg', 'wb') as f:
    f.write(m.system.image_data())
