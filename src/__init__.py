from ._mechanica import *

__all__ = ['forces', 'math']

close = Simulator.close
run = Simulator.run
show = Simulator.show
irun = Simulator.irun
step = Universe.step
stop = Universe.stop
start = Universe.start
Species = carbon.Species
SpeciesList = carbon.SpeciesList

__version__ = _mechanica.__version__

def _plot_potential(p, show=True, ymin=None, ymax=None, *args, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as n

    xx = None

    min = kwargs["min"] if "min" in kwargs else 0
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


    if ymin is None:
        ymin = n.amin(y)

    if ymax is None:
        ymax = n.amax(y)


    plt.ylim(ymin, ymax)
    p = plt.plot(xx, y)

    if show:
        plt.show()

    return p


_mechanica.Potential._set_dict_value("plot", _plot_potential)
