import mechanica as m
import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt

from time import time
from random import random

print ( matplotlib.__version__ )

# potential cutoff distance
cutoff = 10

# number of particles
count = 6000

# number of time points we avg things
avg_pts = 3

# dimensions of universe
dim=np.array([30., 30., 30.])
center = dim / 2

# new simulator
m.Simulator(dim=dim, cutoff=cutoff, integrator=m.FORWARD_EULER, dt=0.002)

clump_radius = 2

# number of bins we use for averaging
avg_bins = 3

prev_pos = np.zeros((count, 3))
avg_vel  = np.zeros((count, 3))
avg_pos  = np.zeros((count, 3))
avg_index = 0;

# the output display array.
# matplotlib uses (Y:X) axis arrays instead of normal (X:Y)
# seriously, WTF matplotlib???
display_velocity = np.zeros((100, 200))
display_velocity_count = np.zeros((100, 200))

# keep a handle on all the cells we've made.
cells = count * [None]

def cartesian_to_spherical(pt, origin):
    """
    Convert a given point in cartesian to a point in spherical (r, theta, phi)
    relative to the origin.
    """

    pt = pt - origin
    r = np.linalg.norm(pt)
    theta = math.atan2(pt[1], pt[0])
    if theta < 0:
        theta = theta + 2 * math.pi
    phi = math.acos(pt[2] / r)
    return np.array((r, theta, phi))


def spherical_index_from_cartesian(pos, origin, theta_bins, phi_bins):
    """
    calculates the i,j indexex in a 2D array of size irange,jrange
    from a given cartesian coordinate and origin.

    theta_bins is number of bins in the theta direction
    phi_bins is number of bins in the phi direction
    """

    # get spherical coordinates in (r, theta, phi)
    sph = cartesian_to_spherical(pos, origin)

    # i is the horizontal axis, theta axis, and j is vertica, the phi
    i = math.floor(sph[1] / (2. * np.pi) * theta_bins)
    j = math.floor(sph[2] / (np.pi + np.finfo(np.float32).eps) * phi_bins)

    return (i, j)


class Yolk(m.Particle):
    mass = 500000
    radius = 6


class Cell(m.Particle):
    mass = 5
    radius = 0.25
    target_temperature=0
    dynamics = m.Overdamped

total_height = 2 * Yolk.radius + 2 * clump_radius
yshift = total_height/2 - Yolk.radius
cshift = total_height/2 - 1.3 * clump_radius


pot_yc = m.Potential.soft_sphere(kappa=300, epsilon=50, r0=1, \
                                 eta=2, tol = 0.03, min=0.1, max=8, shift=True)

pot_cc = m.Potential.soft_sphere(kappa=600, epsilon=0.5, r0=1, \
                                 eta=3, tol = 0.05, min=0, max=2.5, shift=True)

# bind the potential with the *TYPES* of the particles
m.Universe.bind(pot_yc, Yolk, Cell)
m.Universe.bind(pot_cc, Cell, Cell)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = m.forces.random(0, 1)

# bind it just like any other force
m.bind(rforce, Cell)

yolk = Yolk(position=center-[0., 0., yshift])

for i, p in enumerate(m.random_points(m.SolidSphere, count)):
    pos = p * clump_radius + center+[0., 0., cshift]
    cells[i] = Cell(position=pos)


def calc_avg_pos(e):
    global avg_index
    print("calc_avg_pos, index: ", avg_index)

    for i, p in enumerate(cells):
        avg_vel[i] += p.position - prev_pos[i]
        prev_pos[i] = p.position
        avg_pos[i] += p.position

    if avg_index == (avg_bins - 1):

        avg_pos[:] = avg_pos / avg_bins

        for i in range(count):
            # get the theta / phi index from the cartesian coordinate
            # remeber, matplotlib is backwards and wants matricies in
            # transposed order.
            ii, jj = spherical_index_from_cartesian(avg_pos[i], \
                                                    yolk.position, \
                                                    display_velocity.shape[1], \
                                                    display_velocity.shape[0])

            # counts of samples we have for this spherical coordinate
            display_velocity_count[jj, ii] += 1

            # velocity of the vertical (y) direction
            display_velocity[jj, ii] += avg_vel[i][1] / avg_bins

        display_velocity[:] = display_velocity / avg_bins

        Z = display_velocity


        plt.pause(0.01)
        plt.clf()
        #plt.contour(Z)

        yy = np.linspace(0, np.pi, num=100)
        xx = np.linspace(0, 2 * np.pi, num=200)
        c=plt.pcolormesh(xx, yy, Z, cmap ='jet')
        plt.colorbar(c)
        plt.show(block=False)

        avg_pos[:] = 0
        avg_vel[:] = 0
        display_velocity_count[:] = 0
        display_velocity[:] = 0


    # bump counter where we store velocity info to be averaged
    avg_index = (avg_index + 1) % avg_bins

m.on_time(calc_avg_pos, period=0.01)

# run the simulator interactive
m.Simulator.run()
