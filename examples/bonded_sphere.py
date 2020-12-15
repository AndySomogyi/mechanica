import mechanica as m
import numpy as np

m.Simulator(dim=[25., 25., 25.], cutoff=3, dt=0.005, bc=m.BOUNDARY_NONE)

class Blue(m.Particle):
    mass = 10
    radius = 0.05
    dynamics = m.Overdamped
    style = {'color':'dodgerblue'}


class Big(m.Particle):
    mass = 10
    radius = 8
    frozen=True
    style = {'color':'orange'}


# simple harmonic potential to pull particles
h = m.Potential.harmonic(k=200, r0=0.001, max = 5)

#h = m.Potential.linear(k=0.5,  max = 5)

#pb= m.Potential.glj(e=0.00001, r0=0.1, m=3, min=0.01, max=Blue.radius*3)
pb = m.Potential.coulomb(q=0.01, min=0.01, max=3)

# potential between the small and big particles
pot = m.Potential.glj(e=1, m=2, max=5)

Big(m.Universe.center)

Big.style.visible = False

m.bind(pot, Big, Blue)

m.bind(pb, Blue, Blue)


parts, bonds = m.bind_sphere(h, type=Blue, n=4, phi=(0.55 * np.pi, 1 * np.pi), radius = Big.radius + Blue.radius)

#for b in bonds:
#    print("energy(", b.id, "): ", b.energy())


#print("parts: ", parts)


#parts[0].destroy()

#parts[3].destroy()

#parts[7].destroy()



# run the model
m.show()
