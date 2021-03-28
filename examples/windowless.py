import mechanica as m

m.init(windowless=True,
       window_size=[1024,1024],
       clip_planes = [([2, 2, 2], [1, 1, 0]), ([5, 5, 5], [-1, -1, 0])])

print(m.system.gl_info())

class Na (m.Particle):
    radius = 0.4
    style={"color":"orange"}

class Cl (m.Particle):
    radius = 0.25
    style={"color":"spablue"}

uc = m.lattice.bcc(0.9, [Na, Cl])

m.lattice.create_lattice(uc, [10, 10, 10])

# m.system.image_data() is a jpg byte stream of the
# contents of the frame buffer.

with open('system.jpg', 'wb') as f:
    f.write(m.system.image_data())
