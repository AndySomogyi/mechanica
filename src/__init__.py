from ._mechanica import *
from . import lattice

import warnings

__all__ = ['forces', 'math', 'lattice']


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

def _plot_potential(p, s = None, force=True, potential=False, show=True, ymin=None, ymax=None, *args, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as n

    xx = None

    min = kwargs["min"] if "min" in kwargs else 0
    max = kwargs["max"] if "max" in kwargs else p.max
    step = kwargs["step"] if "step" in kwargs else 0.001
    range = kwargs["range"] if "range" in kwargs else (p.min, p.max, step)

    xx = n.arange(*range)

    yforce = None
    ypot = None

    if p.flags & POTENTIAL_SCALED:
        if not s:
            warnings.warn("""plotting scaled function,
            but no 's' parameter for sum of radii given,
            using value of 1 as s""")
            s = 1

        if force:
            yforce = [p(x, s)[1] for x in xx]

        if potential:
            ypot = [p(x, s)[0] for x in xx]

    else:

        if force:
            yforce = [p(x)[1] for x in xx]

        if potential:
            ypot = [p(x)[0] for x in xx]

    if not ymin:
        y = n.array([])
        if yforce:
            y = n.concatenate((y, yforce))
        if ypot:
            y = n.concatenate((y, ypot))
        ymin = n.amin(y)


    if not ymax:
        y = n.array([])
        if yforce:
            y = n.concatenate((y, yforce))
        if ypot:
            y = n.concatenate((y, ypot))
        ymax = n.amax(y)


    yrange = n.abs(ymax - ymin)

    lines = None

    print("ymax: ", ymax, "ymin:", ymin, "yrange:", yrange)

    print("Ylim: ", ymin - 0.1 * yrange, ymax + 0.1 * yrange )

    plt.ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange )

    if yforce and not ypot:
        lines = plt.plot(xx, yforce, label='force')
    elif ypot and not yforce:
        lines = plt.plot(xx, ypot, label='potential')
    elif yforce and ypot:
        lines = [plt.plot(xx, yforce, label='force'), plt.plot(xx, ypot, label='potential')]

    plt.legend()

    plt.title(p.name)

    if show:
        plt.show()

    return lines


_mechanica.Potential._set_dict_value("plot", _plot_potential)
