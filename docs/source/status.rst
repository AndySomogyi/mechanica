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

Version Alpha 1.0.18.1
----------------------
* generalized passive, consumer and producer fluxes
* better OpenGL info reporting, `gl_info()`, `egl_info()`
* enable boundary conditions on chemical speices, bug fix parsing init
  conditions
* use species boundary value to enable source / sinks
* source / sinks in example

Version Alpha 1.0.17.1
----------------------
* multi-threaded rendering fixes

Version Alpha 1.0.16.2
----------------------
* Logging, standardized all logging output, python api for setting log level. 
* fix kinetic energy reporting
* synchronize gl contexts between GLFW and Magnum for multi-thread rendering

Version Alpha 1.0.16.2
----------------------
* initialize Mechanica either via m.init, m.Simulator, or m.simulator.init

Version Alpha 1.0.16.1
----------------------
* finally, completly expunged pybind11! pybind11 is finally GONE!
* context managment methods for multi-threaded headless rendering. 
* universe.reset() method, clears objects
* set window title to script name
* add 'positions()', 'velocities()' and 'forces()' methods to particle list. 
* universe.particles() is now a method, and returns a proper list

Version Alpha 1.0.15.5
----------------------
* bug fix with boundary condition constants

Version Alpha 1.0.15.5
----------------------
* bug fix with force calculation when distance too short: pic random separation
  vector of with minimal distance. Seems to work...
* better diagnostic messages
* added normal to boundary vectors

Version Alpha 1.0.15.4
----------------------
* generalized boundary conditions
* add potentials to boundary conditions
* velocity, free-slip, no-slip and periodic boundary conditions
* render updates, back face culling
* headless rendering, rendering without X11 using GLES on Linux
* generalized power potential
* much improved error handling, much more consistency
* particle list fixes
* Rigid Body Dynamics ! (only cuboids currently supported, but still rigid bodies)
* add potentials to rigid bodies
* python api rigid body updates
* rendering updates, more consistency, simplify
* rigid body particle interactions
* friction force
* more expunging pybind, soon, soon we will be rid of pybind.
* bond dissociation_energy (break strength)
* lattice initializer
* add bonds to lattice initliazer
* performance logging
* updates to dissapative particle dynamics forces
* enable adding DPD force to boundaries. 
* generlized single body force (external force)
* fluid dynamics examples
* visco-elastic materials, with bond breaking
* single-body time-dependent force definitions in python

Version Alpha 1.0.15.2
----------------------
* initial dissapative particle dynamics
* doc constant force, dpd

Version Alpha 1.0.15.1
----------------------


Version Alpha 0.0.14.1
----------------------
* added convenience methods to get spherical and cartesian coords from lists
* updated example models
* update docs
* added plot function in examples to plot polar angle velocity. 
* code cleanup

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



Features to be implemented
--------------------------

* Linux binaries
* :strike:`mouse interction -- rotate, zoom simulation`
* :strike:`Documentation`
* :strike:`Event system to hook up simulation events to user objects`
* :strike: `User definable visualization style`
* :strike:`Nosé–Hoover thermostat`
* :strike:`Destroying particles`
* Collision reactions (when particles collide, they react, and can create and
  destroy particles)
* :strike:`Particle mitois`
* :strike: `attach chemical cargo to particles`
* :strike: `inter-particle flux of chemical cargo`
* reaction-kinetics network at each particle
* :strike: `Windows binaries`
* Movable boundary conditions
* :strike: `reflective boundary conditions (only have periodic now)`
* mouse object picking
* :strike: `Python API for bonded interactions (bonds, angles, dihedrals, impropers)`
* :strike: `pre-made DPD potentials (conservative, friction, thermostat)`
* :strike: `With addition of particle chemical cargo, fluxes and above potentials, we will
  have complete transport-dissapative-particle-dynamics simulation. And
  reactions gives us reactive TDPD.`
* :strike: `Visualization:
  We will attach a 'style' attribute to the particle type that will let users
  define how they're presented in the renderer. This will have attributes such
  as color, size, etc... We want to let users attach transfer functions here,
  that will read particle attributes, such as local chemical concentration and
  map this to a color. To get decent performance, we'll have to compile user
  specified functions into pixel shader and run them on the GPU.`

* :strike: `Multi-process rendering. Jupyter notebooks now uses a two process model, so
  it's probematic to create a window in a background process. We want to enable
  the simulator render to an off-screen buffer, and render to this buffer. Then
  copy the buffer to the foreground process and display it here.`


Known Bugs
----------

* :strike: `In ipython, closing the window does not work correctly`
* :strike: `energy is not conserved -- bug in integrator.`
* only a subset of features are implmented

Features for next release
-------------------------

* Python API for vertex model (polygons, cells)





