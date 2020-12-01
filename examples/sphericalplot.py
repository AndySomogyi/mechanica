import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from time import time
from random import random

print ( matplotlib.__version__ )

class SphericalPlot :


    def __init__(self, cells, origin):

        # keep a handle on all the cells we've made.
        self.cells = cells

        # number of time points we avg things
        self.avg_pts = 3

        # number of bins we use for averaging
        self.avg_bins = 3

        count = len(cells)

        self.prev_pos = np.zeros((count, 3))
        self.avg_vel  = np.zeros((count, 3))
        self.avg_pos  = np.zeros((count, 3))
        self.avg_index = 0;

        # origin of the spherical coordinate system,
        # in cartesion coords
        self.origin = origin


        # the data for the output display array.
        # matplotlib uses (Y:X) axis arrays instead of normal (X:Y)
        # seriously, WTF matplotlib???
        # X (column index) is theta angle
        # Y (row index) is phi angle
        self.data = np.zeros((100, 200))
        #self.display_velocity_count = np.zeros((100, 200))

    def data_index(self, part):
        """
        calculate the index in the output data (self.data) of
        the particle, using the particle's spherical coords
        """

        (r, theta, phi) = part.spherical(self.origin)

        shape = self.data.shape
        # theta goes from [0, 2 pi], phi from [0, pi]
        theta_index = int(shape[1] * (theta / (2. * np.pi)))
        phi_index = int(shape[0] * (phi / (np.pi)))

        return (phi_index, theta_index)


    def update(self, e):

        print("update(", self, ", ", e, ")")

        print("calc_avg_pos, index: ", self.avg_index)

        self.data[:] = 0

        for i, p in enumerate(self.cells):
            (ii,jj) = self.data_index(p)
            self.data[ii,jj] += 1


        plt.pause(0.001)
        plt.clf()
        #plt.contour(Z)

        yy = np.linspace(0, np.pi, num=100)
        xx = np.linspace(0, 2 * np.pi, num=200)
        c=plt.pcolormesh(xx, yy, self.data, cmap ='jet')
        plt.colorbar(c)
        plt.show(block=False)

        # bump counter where we store velocity info to be averaged
        self.avg_index = (self.avg_index + 1) % self.avg_bins
