Mechanica
=========
Mechanica is an interactive particle based physics, chemistry and biology
simulation environment, with a heavy emphasis towards enabling users to model
and simulate complex sub-cellular and cellular biological physics
problems. Mechanica is part of the Tellurium
`<http://tellurium.analogmachine.org>`_ project.

Mechanica is designed first and foremost to enable users to work interactively
with simulations -- so they can build, and run a simulation in real-time, and
interact with that simulation whilst it's running. The goal is to create an
SolidWorks type environment where users can create and explore virtual models of
soft condensed matter physics, with a emphasis towards biological physics.

Mechanica is a native compiled C++ shared library with a native and extensive
Python API, that's designed to used from an ipython console (or via scripts of
course). 

History
=======


Version Alpha 0.0.14
--------------------
* All new FLUX / DIFFUSION / TRANSPORT, We've not got
  Transport-Dissipative-Dynamics working!!!
* secrete methods on particle to perform atomic secrete
* bug fixes in neighbor list, make sure neighbor don't contain the particle
* bug fixes in harmonic potential
* new overlapped sphere potential
* new potential plotting method, lots of nice improvements
* new examples
* update become to copy over species values
* lattice initializers
* add decay to flux
* detect hardware concurrency
* bug fix in Windows release-mode CPUID crash
* multi-threaded integration
* all new C++ thread pool, working on getting rid of OpenMP / pthreads
* event system bug fixes
* documentation updates



Version Alpha 0.0.13
--------------------
* preliminary SBML species per object support
* SBML parsing, create state vector per object
* cpuinfo to determine instruction set support
* neighbor list bug fixes
* improve and simplify events
* on_keypress event
* colormap support per SBML species

Version Alpha 0.0.12
--------------------
* free-slip boundary conditions
* rendering updates
* energy minimizer in initial condition generator
* updates to init condition code
* initial vertex model support


Version Alpha 0.0.11
--------------------
* new linear potential
* triagulated surface mesh generation for spheres, triangulate sphere
  surfaces with particles and bonds, returns the set. 
* banded spherical mesh generation
* bug fixes in making particle list from python list
* points works with spherical geometry
* internal refactoring and updates
* Dynamic Bonds! can dynamically create and destory bonds
* lots of changes to deal with variable bond numbers
* rendering updates for dyanmic bonds
* particle init refactor
* added metrics (pressure, center of mass, etc...) to particle lists
* add properties and methods to Python bond API
* bond energy calcs avail in python
* bond_str and repr
* automatically delete delete bond if particle is deleted

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
