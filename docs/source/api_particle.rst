Particles and Clusters
----------------------


.. class:: Particle(object)

   The particle is the most basic physical object we provide. All other physical
   objects either extend the base Particle, or are collections of particles such
   as the :any:`Cluster`


   .. attribute:: position

      ``vector3`` -- gets / sets the global position of the particle

   .. attribute:: velocity

      ``vector3`` -- returns the velocity of the particle, read-only

   .. attribute:: force

      ``vector3`` -- returns the net force that acts on this particle

   .. attribute:: charge

      ``number`` -- gets the total charge of the particle. 

   .. attribute:: mass


   .. attribute:: radius


   .. attribute:: name


   .. attribute:: name2


   .. attribute:: dynamics

      ``number`` -- one of the :ref:`Integrator Constants` that specifies the
      time evolution of this particle. Particles can be either intertial
      particles that obey Newtonian dynamics, :math:`F=ma` or overdamped
      dynamics, :math:`F \propto mv`. 


   .. attribute:: age


   .. attribute:: style

      ``Style`` -- gets / sets the style of an object. When we create a new
      particle instance, it's style points to the style attribute of the
      particle's type, so that if we change something in the particle type, this
      changes every instance of that type. For more details, see the
      :ref:`style-label` section. 


   .. attribute:: frozen

      Get / sets the `frozen` attribute. Frozen particles are fixed in place,
      and will not move if any force acts on them. 

   .. attribute:: id

   .. attribute:: type_id


   .. attribute:: flags

   .. method:: become(type)

      Dynamically changes the *type* of an object. We can change the type of a
      :any:`Particle` derived object to anyther pre-existing :any:`Particle`
      derived type. What this means is that if we have an object of say type
      *A*, we can change it to another type, say *B*, and and all of the forces
      and processes that acted on objects of type A stip and the forces and
      processes defined for type B now take over. See section :ref:`Changing
      Type` for more details. 

      :param type: (Type) 

   
   .. method:: split()

      Splits a single particle into two, for more details, see section
      :ref:`Splitting and Cleavage`. The particle version of `split` is fairly
      simple, however the :meth:`Cluster.split` offers many more options. 

   .. method:: fission()

      synonym for :meth:`split`

   .. method:: destroy()

      Destroys the particle, and removes it form inventory. The present object
      is handle that now references an empty particle. Calling any method after
      `destroy` will result in an error. 

   .. method:: spherical([origin])

      Calculates the particle's coordinates in spherical coordinates
      (:math:`[\rho, \theta, \phi]`), where :math:`\rho` is the distance from
      the origin, :math:`\theta` is the azimuthal polar angle ranging from
      :math:`[0,2 \pi]`, and :math:`phi` is the declination from vertical, ranging
      from :math:`[0,\pi]`

      :param [x,y,z] origin: a vector of the origin to use for spherical
                             coordinate calculations, optional, if not given,
                             uses the center of the simulation domain as the
                             origin. 

   .. method:: virial([distance])

      Computes the virial tensor, see :ref:`Pressure and Virial Tensors`. 

      :param distance: (number (,optional)) distance from the center of this
                       particle to include the other particles to use for the
                       virial calculation. 

      :rtype: 3x3 matrix


   .. method:: neighbors([distance], [types])

      Gets a list of all the other particles that are near the current one. By
      default, we list all the nearest particles that interact with the current
      one via forces.

      :param distance: (number (,optional)) - An optional search
                   distance, if specified will get all objects within the given
                   distance. Defaults to the global simulation cutoff distance. 

      :param types: (tuple, (,optional)) -- If specified, can provide a tuple
                    of types to include in the neighbor search. If types are
                    provides, this method will return all non-cluster particles
                    within a certain distance. Defaults to all types. 

      For example, to search for all objects of type `A` and `B` a distance of 1
      unit away from a particle `p`, we would::

        >>> nbrs = p.neighbors(distance=1, types=(A, B))
        >>> print(len(nbrs))
  


.. class:: Cluster(Particle)

   A Cluster is a collection of particles.

   .. method:: split([axis], [random], [normal], [point])

      Splits the cluster into two clusters, where the first one is the original
      cluster and the new one is a new 'daughter' cluster.

      split is discussed in detail in :ref:`Splitting and Cleavage`


      :param axis: (length 3 vector (,optional)) - orientation axis for a
                   split. If the 'axis' argument is given, the 'split' method
                   chooses a random plane co-linear with this vector and uses
                   this as the cleavage plane. 

      :param random: (Boolean (,optional)) - 'split' chooses a random cleavage
                     plane coincident with the center of mass of the cluster. 
                  
      :param normal: (length 3 vector (,optional)) - a normal direction for the
                     cleavage plane. 

      :param point: (length 3 vector (,optional)) - if given, uses this point to
                    determine the point-normal form for the clevage plane. 

   .. method:: virial()

      Computes the :ref:`Virial Tensor` for the particles in this cluster. 

   .. method:: radius_of_gyration()

      Computes the :ref:`Radius of Gyration` for the particles in this cluster. 

   .. method:: center_of_mass()

      Computes the :ref:`Center of Mass` for the particles in this cluster. 

   .. method:: center_of_geometry()

      Computes the :ref:`Center of Geometry` for the particles in this cluster. 

   .. method:: moment_of_inertia()

      Computes the :ref:`Moment of Inertia` for the particles in this cluster.


   .. method:: centroid()

      Convenience synonym for :any:`center_of_geometry`

   .. method:: inertia()

      Convenience synonym for :any:`moment_of_inertia`

