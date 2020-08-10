import mechanica as m
import math
import matplotlib.pylab as plt
import numpy as n

h = m.Potential.harmonic_angle(k=10, theta0 = 3.14/2)

xx = n.arange(0, math.pi, 0.1)

co = [math.cos(x) for x in xx]

y = [h(n.cos(x)) for x in xx]

plt.plot(xx, y)

plt.show()
