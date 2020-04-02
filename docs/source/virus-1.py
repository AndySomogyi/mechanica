import mechanica as m
from random import *

# a function that generates a 5-sided particle group, where the group
# is a pentagon composed of 5 paricles and 5 bonds. 
def create_pentamer():
    pass


# create an Argon atom type 
class Argon(m.Partricle):
    mass = 39.94

# explicitly set up initial and boundary conditions

# simulation origin 
origin = [ 0.0 , 0.0 , 0.0 ]

# size of the simulation domain
dim = [ 10.0 , 10.0 , 10.0 ]

# number of particles
n_parts = 32680

# initial teperature
temp = 100.

# with of space partitioning cells. The solver partitions space into a 
# series of boxes, and uses this to optimize long-range force calculations
# and runs one compute thread in each cell (in a queue).
cell_width = 1.0

# long range force cutoff distance. 
cutoff = 1.0

# reference teperatrue
Temp = 100.0;

# explicitly initialize the universe
m.Universe(origin=origin, 
           dim=dim,
           spacecell_dim = [cell_width, cell_width, cell_width],
           cutoff=cutoff)

# Create a 12-6 inter-atom potential
# The smallest radius for which the potential will be constructed.
# The largest radius for which the potential will be constructed.
# The first parameter of the Lennard-Jones potential.
# The second parameter of the Lennard-Jones potential.
# The tolerance to which the interpolation should match the exact
# potential.
pot_ArAr = m.Potential.LJ126(range=(0.275 , 1.0), A=9.5075e-06 , B=6.1545e-03 , tol=1.0e-3 ) 

# bind the potential with Argon type
m.bind(pot_ArAr, Argon)

# create a lattice of particles
# these are number of particles in each dimension. 
nx = ceil( pow( nr_parts , 1.0/3 ) ); hx = dim[0] / nx;
ny = ceil( sqrt( nr_parts / nx  ));   hy = dim[1] / ny;
nz = ceil( nr_parts / nx / ny );      hz = dim[2] / nz;

# iterate over the how many particles we want, set thier
# initial positions and velocities, and add them to the 
# universe. 
for i in range(0, nx):
    x[0] = 0.05 + i * hx
    for  j in range(0, ny):
        x[1] = 0.05 + j * hy
        for k in range(0, nz):            
            x[2] = 0.05 + k * hz;
            
            # velocity uniform random
            v = np.random.rand(3,1) - 0.5
            temp = 0.275 / np.linalg.norm(v)
            v *= temp
            vtot[0] += pAr.v[0]; vtot[1] += pAr.v[1]; vtot[2] += pAr.v[2];
            
            # create a new Argon, implicitly add it to the universe
            Argon(pos=x, vel=v)


# create a rendering window and renderer
# the visible window is a generic output device
# by default, a new window will not display anything until
# it has a some renderers. 
win = m.Window(width=640, height=480)

# renderer is responsible for displaying the contents of the universe
# in the window. We have multiple different kinds of renderers for
# different kinds of content, such as network graphs...
# 
# If no args are given to renderer (such as what 'renderable' this to display
# it defaults to the universe renderer. 
m.Renderer(win)


# we attach an event handler to the on_step event to display 
# stats every 100 time steps. The solver will call this method
# perioically. 
def print_stats(u):
     #get the total COM-velocities and ekin
    epot = Universe.potentialEnergy
    ekin = 0.0;
    v = 0
    
    # iterate over every particle in the universe, and grab it's 
    # kinetic energy. 
    for p in Universe.particles:
        v += np.linalg.norm(p.vel)
        ekin += 0.5 * 39.948 * v2;
        
    # compute the temperature and scaling
    temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * len(Universe.particles) )
    w = sqrt( 1.0 + 0.1 * ( Temp / temp - 1.0 ) );
    
    printf("time: %f pot energy: %e, kinetic energy: %e, temperature: %e,"
           "swaps: %i,  stalls: %i" %
           u.time, epot, ekin, temp, u.swaps, u.stalls)

    
# set the number of worker threads, 
Universe.threads = 2
Universe.dt = 0.005
Universe.steps = 5000
    
    
# add the print stats function to the on step event handler. 
Universe.on_step(100) += print_stats

# run the universe in async mode. This means that this method will kick off the 
# worker threads, run the simulation, imedietly return. 
Universe.run(async=True)




























