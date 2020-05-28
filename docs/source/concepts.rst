Key Concepts
============


Running a Simulation
--------------------

In most cases, running simulation is as simple as initializing the
:class:`Simulator`, building the physical model, and calling either the
:meth:`Simulator.run` or :meth:`Simulator.irun` methods to run the
simulation. However sometimes users might want more control. 

A running simulation has two key components: interacting with the operating
system, and listening for user input, and time stepping the simulation (model)
itself. Whenever the :meth:`Simulator.run` or :meth:`Simulator.irun` are
invoked, they automatically start time-stepping the simulation. Users can
however explicitly control the time-stepping of the model directly. To display
the window, and start the operating system message loop, you can call the
:meth:`Simulator.show` method. This works just like the MatPlotLib show, in that
it displays the windows, but does not time step the simulation. The
:meth:`Universe.start`, :meth:`Universe.step`, :meth:`Universe.stop` methods
start the universe time evolution, perform a single time step, and stop the time
evolution. If the universe is stopped, you can simplly call the
:meth:`Universe.start` method to continue where it was stopped. All methods to build
and manipulate the universe are available either with the universe stopped or
running.



Building A Model
----------------


The first step in formalizing knowledge is writing it down in such a way that it
has semantic meaning for both humans and computers. This section will cover the
key concepts Mechanica provides that enable users to build models of physical
things.

The two key concepts we cover here are *objects* and *processes*. Objects are
logical representations of physical matter. We use a particle to represent
either individual things such as cells, molecules, etc, or particles can
represent clumps of matter, such a volume of fluid. Processes are the ways
objects interact with each other. Here we will cover the basic interaction
potentials that we provide, and will also cover reactions, fluxes and events.


Making Things Move
------------------

Newton's first law states that an object either remains at rest or continues to
move at a constant velocity, unless acted upon by a force. That is true
regaurdless if we are considering atoms or galaxies. In order make any object in
Mechanica move, we must apply a force to it. To make objects move, Mechanica
sums up all of the forces that act on an object, and uses that to calculte the
object's velocity and position. 

The nature of forces in Mechanica are *incredibly* flexible, but we provide a
variety of built-in forces to enable common behaviors.

Conservative forces are usually a kind of :class:`Potential` object, where the
force is described in terms of it's potential energy function. Long-range,
fluid, and most bonded interactions are examples of conservative potential
energy fuction based forces. All potential based forces contibute to the total
potential energy of the system, and we can read the total potential energy
either via the :attr:`Universe.potential_energy` attribute, or we can also read
the potential energy of all objects of a type, via the type's
``potential_energy`` attribute.

We make it easy to create forces, and apply them to objects::

  # create a potential, for a simple lennard-jones fluid: 
  fluid_potential = Potential.lennard-jones-12-6(…)

  # bind it to ALL types
  m.bind(fluid_potential, Particle, Particle) 

This example creates a simple potential, and binds it to ALL objects. As all
objects in our modeling world are either an instance of the base ``Particle``
type, or a instance of a subclass of it.



Controlling Temperature
-----------------------




Right now, I have the concept of a ‘Potential’, these are objects that are specified in terms of potential function, and internally, the integrator does a bit of magic with them, and uses them calculate the conservative force that gets added to the total force. Things like bonds, angles, long-range non-bonded forces are all specified in terms of potentials. This works great for conservative forces, and is numerically actually faster then specifying a force function directly. Also, but specifying conservative forces as a potential, that lets me have both a ‘potential_energy’ and a ‘kinetic_energy’ attributes on the universe (and also the object type, i.e. if a user creates a ‘MyParticleType’, they can call MyParticleType.kinetic_energy and this returns the total kinetic energy of all objets of this type).  

However, for non-conservative forces, like temperature, friction, etc, these are almost always defined as forces. We can associate a potential energy with a conservative force, but not a non-conservative (or random) force.

That would imply that we need have to allow the user to represent both potentials and forces. I would have preferred to just work in potential or forces, as this simplifies the things for the users, but I don’t really see a way around it. 

So, user experience would be like this:

# create a thermostat force, effectively maintains the temperate of a set of things
thermostat = Force.langevin_thermostat(298)

# bind it to all objects of type MyParticle
m.bind(thermostat, MyParticle)

# create a friction force
friction = Force.friction(…)

# bind it to all objects of type SomeOtherParticle
m.bind(friction, SomeOtherParticle)



 .. _binding:

Binding
-------

Binding objects and processes together is one of the key ways to create a
Mechanica simulation. Binding is a very generic concept, but essentially it
serves to connect a process (such a potential, flux, reaction, etc..) with one
or more objects that that process should act on.

Binding Families of Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Binding Individual Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^

Particles
---------

Universe
--------

Potentials
----------

Conservative Potentials
^^^^^^^^^^^^^^^^^^^^^^^

Random Potentials
^^^^^^^^^^^^^^^^^

Dissipative Potentials
^^^^^^^^^^^^^^^^^^^^^^

Collisions / Reactions
----------------------

Adjective Materials / Diffusive / Dissolved Chemicals
-----------------------------------------------------

Interacting With The Operating System
-------------------------------------

The :class:`Simulator` is manages all of the interaction between the operating
system, end user input, external messaging and the physical model (which resides
in the :class:`Universe` object.

In order to display a window (s), receive user input, and listen for external
messages, the simulator needs to run an *event loop*. These are handled by the
:meth:`Simulator.run` and :meth:`Simulator.irun` methods. 

Types and Subtypes
------------------




