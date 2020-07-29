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
* new dynamcis options, include both Newtonian (Velocity-Verlet) and
  overdamped. 
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
* major bug fix in not making sure potenal distance does not go past end of
  interpolation segments.
* new random force
* new soft-sphere interaction potential
* add radius to particle type def
* update renderer to draw different sized particles
* add number of space cells to simulator constructor
* configurable dynamics (Newtonain, Overdamped), more to come
  particle delete functionality, and fix particle events
* examples bind events to destory, creation and mitosis methods
* new event model 

Version Alpha 0.0.1 Dev 3
-------------------------

* Refactoring of Particle python meta-types, simpler and cleaner
* Upgrade to GLFW 3.3
* New single body generalized force system
* Berendsen thermostat as first example single body generalized forces
* Per-type thermostat
* Arc-ball user interaction
* Simplify and eliminate redundency between C++ and Python apps. 


Version Alpha 0.0.1 Dev 2
-------------------------
* First public release
