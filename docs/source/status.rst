Status
======

Mechancia is very early in the development cycle, as such, it's *EXTREMLY*
feature-incomplete, unstable, and will almost certainly crash.

However, as such, this is YOUR chance to try it out, and let us know what kind
of features you'd like to see.

Features to be implemented
--------------------------

* Linux binaries
* mouse interction -- rotate, zoom simulation
* Documentation
* Event system to hook up simulation events to user objects
* User definable visualization style
* Nosé–Hoover thermostat
* Destroying particles
* Collision reactions (when particles collide, they react, and can create and
  destroy particles)
* Particle mitois
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





