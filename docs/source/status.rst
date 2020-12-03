.. _status:

.. role:: strike
    :class: strike

Status
======

Mechancia is very early in the development cycle, as such, it's *EXTREMLY*
feature-incomplete, unstable, and will almost certainly crash.

However, as such, this is YOUR chance to try it out, and let us know what kind
of features you'd like to see.


History
=======

Version Alpha 0.0.10-dev1
-------------------------
* bug fixes in bond pairwise search
* improved particle `__repr__`, `__str__`
* new `style` visible attribute to style to toggle visibility on any 
  rendered object
* make show() work in command line mode
* internal changes for more consistent use of handles vs direct pointers
* `bind_pairwise` to search a particle list for pairs, and bind them with a
  bond.
* new `points` and `random_points` to generate position distributions
* spherical plot updates
* new `distance` method on particles
* implmement `become`  -- now allow dynamic type change
* big fixes in simulation start right away instead of wait for event
* basic bond rendering (still lines, will upgrade to cylinders in future
* render large particles with higher resolution
* new particle list composite structure, all particles returned
  to python in this new list type. fast low overhead list.
* major performance improvment, large object cutoff optimization
* numpy array conversion bug fix
* neighbor list for particles in range
* enumerate all particles of type with 'items()'
* new c++ <-> python type conversions, getting rid of pybind.
* better error handling, check space cells are compatible with periodic boundary
  conditions.
* add `start`, `stop`, `show`, etc. methods to top-level as convenience.
* fix ipython interaction with `show`, default is universe not running when showing
* enable single stepping and visualization with ipython
* enable start and stop with keyboard space bar. 
* pressure tensor calculations, add to different objects.
* new `Universe.center` property
* better error handling in `Universe.bind`
* clean up of importing numpy
* expose periodic boundary conditions to python.
* periodic on individual axis.
* new metrics calculations, including center of mass, radius of gyration,
  centroid, moment of inertia
* new spherical coords method
* frozen particles
* add harmonic term to generalized Lennard-Jones 'glj' potential

Version Alpha 0.0.9-dev4
------------------------
* tweaks in example models
* more options (periodic, max distance) in simulator ctor
* add flags to potentials
* persistence time in random force
* frozen option for particles
* make glj also have harmonic potential
* in force eval, if distance is less than min, set eval force to value at min position.
* accept bound python methods for events

Version Alpha 0.0.9
-------------------
* all new cluster dynamics to create sub-cellular element models
* cluster splitting
* splitting via cleavage plane
* splitting via cleavage axis
* other splitting options
* new potential system to deal with cluster and non-cluster interactions
* revamped generalized Lennard-Jones (glj) potential
* new 'shifted' potential takes into account particle radius
* updated potential plotting
* more examples
* fixed major integrator bug

Version Alpha 0.0.8
-------------------
* explicit Bond and Angle objects 
* new example apps 
* new square well potential to model constrained particles
* bug fixes in potential
* thread count in Simulator init


Version Alpha 0.0.7
-------------------
* lots of changes related to running in Spyder. 
* force windows of background process to forground
* detect if running in IPython connsole -- use different message loop
* fix re-entrancy bugs in ipython message loop. 
* Spyder on Windows tested. 

Version Alpha 0.0.6
-------------------
* lots of changes to simulation running / showing windows / closing windows, etc..
* documentation updates

Version Alpha 0.0.5 Dev 1
-------------------------

* Add documentation to event handlers, and example programs
* fix bugs in creating event events 
* add version info to build system and make available as API. 


Version Alpha 0.0.4 Dev 1
-------------------------
* All new particle rendering based on instanced meshes. Rendering quality is
  dramatically improved. Now in a position to do all sorts of discrete elements
  like ellipsoids, bonds, rigid particles, etc... 
* Implement NOMStyle objects. This is essentially the CSS model, but for 3D
  applications. Each object has a 'style' property that's a collection of all
  sorts of style attributes. The renderer looks at the current object, and chain
  of parent objects to find style attributes. Basically the CSS approach. 
* More demo applications. 
* Memory bugs resolved. 

Version Alpha 0.0.3 Dev 1
-------------------------
* Windows Build! 
* lots of portability updates
* some memleak fixes

Version Alpha 0.0.2 Dev 5
-------------------------

* lots of new documentation
* reorganize utility stuff to utily file
* add performance timing info to particle engine
* add examples (multi-size particles, random force, epiboly, 
  events with creation, destruction, mitosis, ...)
* new dynamics options, include both Newtonian (Velocity-Verlet) and
  over-damped. 
* new defaults to set space cell size, better threading
* New explicit bond object
* add creation time / age to particle
* particle fission (mitosis) method (simple)
* clean up potential flags
* harmonic potential
* new reactive potential to trigger (partial implementation)
* random points function to create points for geometric regions
* prime number generator
* Fixed major bug in cell pair force calculation (was in wrong direction)
* major bug fix in not making sure potential distance does not go past end of
  interpolation segments.
* new random force
* new soft-sphere interaction potential
* add radius to particle type def
* update renderer to draw different sized particles
* add number of space cells to simulator constructor
* configurable dynamics (Newtonian, Over-damped), more to come
  particle delete functionality, and fix particle events
* examples bind events to destroy, creation and mitosis methods
* new event model 

Version Alpha 0.0.1 Dev 3
-------------------------

* Refactoring of Particle python meta-types, simpler and cleaner
* Upgrade to GLFW 3.3
* New single body generalized force system
* Berendsen thermostat as first example single body generalized forces
* Per-type thermostat
* Arc-ball user interaction
* Simplify and eliminate redundancy between C++ and Python apps. 


Version Alpha 0.0.1 Dev 2
-------------------------
* First public release



Features to be implemented
--------------------------

* Linux binaries
* :strike:`mouse interction -- rotate, zoom simulation`
* Documentation
* :strike:`Event system to hook up simulation events to user objects`
* User definable visualization style
* :strike:`Nosé–Hoover thermostat`
* :strike:`Destroying particles`
* Collision reactions (when particles collide, they react, and can create and
  destroy particles)
* :strike:`Particle mitois`
* attach chemical cargo to particles
* inter-particle flux of chemical cargo
* reaction-kinetics network at each particle
* Windows binaries
* Movable boundary conditions
* reflective boundary conditions (only have periodic now)
* mouse object picking
* Python API for bonded interactions (bonds, angles, dihedrals, impropers)
* pre-made DPD potentials (conservative, friction, thermostat)
* With addition of particle chemical cargo, fluxes and above potentials, we will
  have complete transport-dissapative-particle-dynamics simulation. And
  reactions gives us reactive TDPD.
* Visualization:
  We will attach a `style` attribute to the particle type that will let users
  define how they're presented in the renderer. This will have attributes such
  as color, size, etc... We want to let users attach transfer functions here,
  that will read particle attributes, such as local chemical concentration and
  map this to a color. To get decent performance, we'll have to compile user
  specified functions into pixel shader and run them on the GPU.
* Multi-process rendering. Jupyter notebooks now uses a two process model, so
  it's probematic to create a window in a background process. We want to enable
  the simulator render to an off-screen buffer, and render to this buffer. Then
  copy the buffer to the foreground process and display it here. 


Known Bugs
----------

* In ipython, closing the window does not work correctly
* energy is not conserved -- bug in integrator.
* only a subset of features are implmented

Features for next release
-------------------------

* Python API for vertex model (polygons, cells)





