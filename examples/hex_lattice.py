import mechanica as m

m.Simulator()


class A (m.Particle):
    radius = 0.2

uc = m.lattice.hex(1, A)

print(uc)



m.lattice.create_lattice(uc, [6, 4, 6])

m.show()
