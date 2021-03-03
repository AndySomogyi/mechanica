ParticleList
------------


.. class:: ParticleList

   Most Mechanica functions that return a list of particle return a
   ParticleList. This is a special list type that adds a lot of convenience
   methods for dealing with spatial information. 

   .. method:: virial()

      Returns the virial, or pressure tensor for all objects in the list.

   .. method:: radius_of_gyration()

      Computes the radius of gyration

   .. method:: center_of_mass()

      Computes the center of mass.

   .. method:: center_of_geometry()

      Computes the center of geometry

   .. method:: centroid()

      Computes the centroid

   .. method:: moment_of_inertia()

      Computes the moment of inertia tensor

   .. method:: inertia()

      Computes the moment of inertia tensor, a synonum for
      :meth:`moment_of_intertia`

   .. method:: copy()

      Creates a deep copy of this list.

   .. method:: positions()

      Returns a numpy 3xN array of all the particle positions. 

   .. method:: spherical_positions([origin])

      Computes the positions in spherical coordinates relative to an origin,
      returns a 3xN array. 

   .. method:: velocities()

      Returns a 3xN array of particle velocities

   .. method:: forces()

      Returns a 3xN array of particle forces. 
