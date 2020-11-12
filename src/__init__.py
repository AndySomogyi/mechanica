from ._mechanica import *

__all__ = ['forces', 'math']

close = Simulator.close
run = Simulator.run
show = Simulator.show
irun = Simulator.irun

__version__ = _mechanica.__version__

def _plot_potential(p, show=True, *args, **kwargs):
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


    miny = n.amin(y)
    plt.ylim(1.1*miny, -5*miny)
    p = plt.plot(xx, y)

    if show:
        plt.show()

    return p


_mechanica.Potential._set_dict_value("plot", _plot_potential)
