from ._mechanica import *

__all__ = ['forces', 'math']

close = Simulator.close
run = Simulator.run
show = Simulator.show
irun = Simulator.irun


def plot_potential(p, *args, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as n

    xx = None

    min = kwargs["min"] if "min" in kwargs else p.min
    max = kwargs["max"] if "max" in kwargs else p.max
    step = kwargs["step"] if "step" in kwargs else (max-min)/1000.
    range = kwargs["range"] if "range" in kwargs else (min, max, step)

    xx = n.arange(*range)

    if p.flags & POTENTIAL_SCALED:
        ri = kwargs["ri"]
        rj = kwargs["rj"]

        y = [p(x, ri, rj) for x in xx]

    else:
        y = [p(x) for x in xx]

    plt.plot(xx, y)
    plt.show()
