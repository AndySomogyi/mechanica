import mechanica as m
import numpy as np

# dimensions of universe
dim=np.array([30., 30., 30.])
center = dim / 2

m.init(dim=dim,
            cutoff=3,
            integrator=m.FORWARD_EULER,
            dt=0.001,
            cells=[6, 6, 6])

class C(m.Cluster):
    radius=5

    class B(m.Particle):
        radius=0.25
        dynamics = m.Overdamped
        mass=15
        style={"color":"skyblue"}

    def split(self, event):
        print("splitting cluster, C.split(" + str(self) + ", event: " + str(event) + ")")
        axis = self.position - C.yolk_pos
        print("axis: " + str(axis))
        m.Cluster.split(self, axis=axis)

        print("new cluster count: ", len(C.items()))



m.on_time(C.split, period=1, predicate="largest")

class Yolk(m.Particle):
    radius = 10
    mass = 1000000
    dynamics=m.Overdamped
    flozen=True
    style={"color":"gold"}

total_height = 2 * Yolk.radius + 2 * C.radius
yshift = 1.5 * (total_height/2 - Yolk.radius)
cshift = total_height/2 - C.radius - 1

yolk = Yolk(position=center-[0., 0., yshift])

c = C(position=yolk.position + [0, 0, yolk.radius + C.radius - 5])

C.yolk_pos = yolk.position

c.B(8000)

pb  = m.Potential.morse(d=15, a=5.5, min=0.1, max=2)
pub = m.Potential.morse(d=1, a=6, min=0.1, max=2)
py = m.Potential.morse(d=3, a=3, max=30)

rforce = m.forces.random(0, 500, 0.0001)

m.bind(rforce, C.B)
m.bind(pb, C.B, C.B, bound=True)
m.bind(pub, C.B, C.B, bound=False)
m.bind(py, Yolk, C.B)

print("initial cluster count: ", len(C.items()))

m.irun()
