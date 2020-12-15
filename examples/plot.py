import mechanica as m

radius = 0.01

p = pb   = m.Potential.glj(e=0.00001, r0=0.1, m=3, min=0.01, max=radius*3)

p.plot()
