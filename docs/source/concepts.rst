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




