import mechanica as m
import numpy as np

m.Simulator(dim=[25., 25., 25.], cutoff=3, bc=m.BOUNDARY_NONE)

class Blue(m.Particle):
    mass = 1
    radius = 0.1
    dynamics = m.Overdamped
    style = {'color':'dodgerblue'}


class Big(m.Particle):
    mass = 1
    radius = 8
    frozen=True
    style = {'color':'orange'}


# simple harmonic potential to pull particles
pot = m.Potential.harmonic(k=1, r0=0.1, max = 3)

# make big cell in the middle
Big(m.Universe.center)

#Big.style.visible = False

# create a uniform mesh of particles and bonds on the surface of a sphere
parts, bonds = m.bind_sphere(pot, type=Blue, n=5, phi=(0.6 * np.pi, 0.8*np.pi), radius = Big.radius + Blue.radius)

# run the model
m.show()
