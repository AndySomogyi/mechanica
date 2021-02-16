import mechanica as m
import numpy as np



# dimensions of universe
dim=np.array([30., 30., 30.])

dist = 3.9

offset = 6

m.Simulator(dim=dim,
            cutoff=7,
            cells=[3,3,3],
            integrator=m.FORWARD_EULER,
            dt=0.01,
            bc={'z':'potential', 'x' : 'potential', 'y' : 'potential'})

class A(m.Particle):
    radius=1
    dynamics = m.Newtonian
    mass=2.5
    style={"color":"MediumSeaGreen"}

class Sphere(m.Particle):
    radius=3
    frozen = True
    style={"color":"orange"}

class Test(m.Particle):
    radius=0
    frozen = True
    style={"color":"orange"}


p = m.Potential.glj(e=50, m=2, max=5)

m.bind(p, A, Sphere)
m.bind(p, A, Test)
m.bind(p, A, m.Cuboid)

m.bind(p, A, m.Universe.boundary_conditions.bottom)
m.bind(p, A, m.Universe.boundary_conditions.top)
m.bind(p, A, m.Universe.boundary_conditions.left)
m.bind(p, A, m.Universe.boundary_conditions.right)
m.bind(p, A, m.Universe.boundary_conditions.front)
m.bind(p, A, m.Universe.boundary_conditions.back)


# above the sphere
Sphere(m.Universe.center + [5, 0, 0])
A(m.Universe.center + [5, 0, Sphere.radius + dist])

# above the test
Test(m.Universe.center + [6, -6, 6])
A(m.Universe.center + [6, -6, 6+dist])

# above the cube
c = m.Cuboid(m.Universe.center + [-5, 0, 0], size=[6, 6, 6])
A(m.Universe.center + [-5, 0, 3 + dist])

# bottom of simulation
A([m.Universe.center[0], m.Universe.center[1], dist])

# top of simulation
A([m.Universe.center[0], m.Universe.center[1], dim[2]-dist])

# left of simulation
A([dist, m.Universe.center[1] - offset , m.Universe.center[2]])

# right of simulation
A([dim[0] - dist, m.Universe.center[1] + offset , m.Universe.center[2]])

# front of simulation
A([m.Universe.center[0], dist , m.Universe.center[2]])

# back of simulation
A([m.Universe.center[0], dim[1] - dist , m.Universe.center[2]])





m.run()
