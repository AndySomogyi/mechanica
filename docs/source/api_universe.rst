Universe
--------



.. class:: Universe(object)

   The universe is a top level singleton object, and is automatically
   initialized when the simulator loads. The universe is a representation of the
   physical universe that we are simulating, and is the repository for all
   phyical object representations.

   All properties and methods on the universe are static, and you never actually
   instantiate a universe.

   Universe has a variety of properties such as boundary conditions, and stores
   all the physical objects such as particles, bonds, potentials, etc..

   .. staticmethod:: bind(thing, a, b)




   .. attribute:: kinetic_energy

      A read-only attribute that returns the total kinetic energy of the system

      :type: double

   .. attribute:: dt

      Get the main simulation time step.

      :type: double

   .. attribute:: time

      Get the current simulation time

   .. attribute:: temperature

      Get / set the universe temperature.

      The universe can be run with, or without a thermostat. With a thermostat,
      getting / setting the temperature changes the temperture that the thermostat
      will try to keep the universe at. When the universe is run without a
      thermostat, reading the temperature returns the computed universe temp, but
      attempting to set the temperature yields an error. 

   .. attribute:: boltzmann_constant

      Get / set the Boltzmann constant, used to convert average kinetic energy to
      temperature


   .. attribute:: particles

      List of all the particles in the universe. A particle can be removed from the
      universe using the standard python ``del`` syntax ::
      
        del Universe.particles[23]

      :type: list

   .. attribute:: dim

      Get / set the size of the universe, this is a length 3 list of
      floats. Currently we can only read the size, but want to enable changing
      universe size.

      :type: Vector3


   


