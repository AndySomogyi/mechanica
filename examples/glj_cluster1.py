import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])
center = dim / 2

m.Simulator(example="",
            dim=dim,
            cutoff=12,
            integrator=m.FORWARD_EULER,
            dt=0.0005)

class C(m.Cluster):
    radius=10

    class B(m.Particle):
        radius=0.5
        dynamics = m.Overdamped
        mass=10
        style={"color":"skyblue"}

c = C(position=center)

c.B(2000)

pb  = m.Potential.glj(e=5, m=3)

rforce = m.forces.random(0, 50)

m.bind(rforce, C.B)
m.bind(pb, C.B, C.B, bound=True)

m.Simulator.irun()
