import mechanica as m
from random import *


# create a boids system with a single particle type. 
class MyBoid(m.VertexCell):
    pass


# Flocking behavior: in addition to other applications, the separation, 
# cohesion and alignment behaviors can be combined to produce the boids 
# model of flocks, herds and schools

# we can add these force processed to particle types

# Separation steering behavior gives a agent the ability to maintain a 
# certain separation distance from others nearby. This can be used to prevent 
# agents from crowding together. Separation Pushes boids apart to keep them 
# from crashing into each other by maintaining distance from nearby flock 
# mates. Each boid considers its distance to other flock mates in its 
# neighborhood and applies a repulsive force in the opposite direction, 
# scaled by the inverse of the distance.

# Implement the separation by adding a scaled Coulomb force to the particle type
# The Colulomb is a 1/r potential with a k constant, set that constant here 
# as 0.01. Calling the forces constructor, with a pair of types automaticaly 
# registers this force with the runtime. 
#
# The last argument here is the type of the object we want to attach to. The 
# Coulomb is a two-body potential, and when only one type is given, the 
# potential is applied to all pairs of this type. 
m.forces.Coulomb(0.01, type(MyBoid))

# Cohesion Keeps boids together as a group. Each boid moves in the 
# direction of the average position of its neighbors. We Compute the direction 
# to the average position of local flock mates and steer in that direction.
# The boids cohesion force is usally implemented as a Hookean force to the 
# average position of all the boid's neighbors. We have a built-in force that 
# implments this behavior. 
m.forces.BoidsCohesion(cutoff=20, k=0.5, type(MyBoid))

# Alignment Drives boids to head in the same direction with similar velocities 
# (velocity matching). Calculate average velocity of flock mates in neighborhood 
# and steer towards that velocity. The alignment force is usually implmented as
# a hookean force between a boid's velocity and it's neightbors velocity. 
m.forces.BoidsAlignment(cutoff = 20, k=0.24, type(MyBoid))


# create a rendering window and renderer
# the visible window is a generic output device
win = m.Window(width=640, height=480)

# renderer is responsible for displaying the contents of the universe
# in the window. We have multiple different kinds of renderers for
# different kinds of content, such as network graphs...
m.Renderer(win)

# create an optional universe interactor, this maps user mouse input to 
# universe objects. 
m.Interactor(win)

#create 50 new particles
for i in range(50):
    # universe extents, a 3x2 matrix
    ex = Universe.extents
    MyBoid(pos=[uniform(ex[0,0], ex[0,1]), 
                       uniform(ex[1,0], ex[1,1]), 
                       uniform(ex[2,0], ex[2,1])])
    
    
    
# run the universe 
Universe.run()





















