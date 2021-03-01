import mechanica as m

m.Logger.set_level(m.Logger.LOG_TRACE)

def plot_glj(r0, e=1, rr0=1, _m=2, n=3):
    p = m.Potential.glj(e=e, r0=rr0, m=_m, n=n, min=0.01, max=10)

    p.plot(s=r0, min=0.8, potential=True, force=True, ymin=-10, ymax=1)


def plot_glj2(r0, e=1, rr0=1, _m=2, n=3):

    p = m.Potential.glj(e=30, m=4, n=2, max=10)

    p.plot(s=r0, min=0.5, max=2, potential=False, force=True)

def plot_glj3(r0, e=1, rr0=1, _m=2, n=3):

    p = m.Potential.glj(e=1, r0=1, m=3, k=0, min=0.1, max=1.5*20, tol=0.1)

    p.plot(s=r0, min=19, max=21, potential=False, force=True)





radius = 0.01

#p = pb   = m.Potential.glj(e=0.00001, r0=0.1, m=3, min=0.01, max=radius*3)

#p.plot()

plot_glj3(20)
