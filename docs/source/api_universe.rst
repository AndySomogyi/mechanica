Universe
--------


.. class:: universe

   The universe is a top level singleton object, and is automatically
   initialized when the simulator loads. The universe is a representation of the
   physical universe that we are simulating, and is the repository for all
   physical object representations.

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
      getting / setting the temperature changes the temperature that the thermostat
      will try to keep the universe at. When the universe is run without a
      thermostat, reading the temperature returns the computed universe temp, but
      attempting to set the temperature yields an error. 

   .. attribute:: boltzmann_constant

      Get / set the Boltzmann constant, used to convert average kinetic energy to
      temperature


   .. staticmethod:: particles()

      List of all the particles in the universe. A particle can be removed from the
      universe using the standard python ``del`` syntax ::
      
        del universe.particles()[23]

      :type: list

   .. attribute:: dim

      Get / set the size of the universe, this is a length 3 list of
      floats. Currently we can only read the size, but want to enable changing
      universe size.

      :type: Vector3


   .. staticmethod:: start()

      Starts the universe time evolution, and advanced the universe forward by
      timesteps in ``dt``. All methods to build and manipulate universe objects
      are valid whether the universe time evolution is running or stopped.

   .. staticmethod:: stop()

      Stops the universe time evolution. This essentially freezes the universe,
      everything remains the same, except time no longer moves forward.

   .. staticmethod:: step(until=None, dt=None)

      Performs a single time step ``dt`` of the universe if no arguments are
      given. Optionally runs until ``until``, and can use a different timestep
      of ``dt``.

      :param until: runs the timestep for this length of time, optional.
      :param dt: overrides the existing time step, and uses this value for time
                 stepping, optional.

   .. staticmethod:: grid(shape)

      Gets a three-dimesional array of particle lists, of all the particles in
      the system. Each

      :param shape: (length 3 array), list of how many grids we want in the x,
                    y, z directions. Minimum is [1, 1, 1], which will return a
                    array with a single list of all particles. 



   .. staticmethod:: virial([origin], [radius], [types])

      Computes the :ref:`Virial Tensor` for the either the entire simulation
      domain, or a specific local virial tensor at a location and
      radius. Optionally can accept a list of particle types to restrict the
      virial calculation for specify types.

      :param origin: An optional length-3 array for the origin. Defaults to the
                     center of the simulation domain if not given.

      :param radius: An optional number specifying the size of the region to
                     compute the virial tensor for. Defaults to the entire
                     simulation domain.

      :param types: An optional list of :class:`Particle` types to include in
                    the calculation. Defaults to every particle type. 

