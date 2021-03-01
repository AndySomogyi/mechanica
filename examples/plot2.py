import mechanica as m

m.Logger.set_level(m.Logger.LOG_DEBUG)

def plot_glj(r0, e=1, rr0=1, _m=2, n=3):
    p = m.Potential.glj(e=e, r0=rr0, m=_m, n=n, min=0.01, max=10)

    p.plot(s=r0, min=0.8, potential=True, force=True, ymin=-10, ymax=1)


def plot_glj2(r0, e=1, rr0=1, _m=2, n=3):

    p = m.Potential.glj(e=30, m=2, max=10)

    p.plot(s=r0, min=0.0, max=1, potential=True, force=True, ymin=-1000000000, ymax=1000000)




radius = 0.01

#p = pb   = m.Potential.glj(e=0.00001, r0=0.1, m=3, min=0.01, max=radius*3)

#p.plot()

plot_glj2(4)
