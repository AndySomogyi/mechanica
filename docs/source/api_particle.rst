Particles and Clusters
----------------------




.. class:: Particle(object)

   The particle is the most basic physical object we provide. All other physical
   objects either extend the base Particle, or are collections of particles such
   as the :any:`Cluster`


   .. attribute:: position


   .. attribute:: velocity


   .. attribute:: force


   .. attribute:: charge


   .. attribute:: mass


   .. attribute:: radius


   .. attribute:: name


   .. attribute:: name2


   .. attribute:: dynamics


   .. attribute:: age


   .. attribute:: style


   .. attribute:: frozen

   .. attribute:: id

   .. attribute:: type_id


   .. attribute:: flags

   .. method:: fission()

   .. method:: split()


   .. method:: destroy

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

   .. method:: pressure()




.. class:: Cluster(Particle)

   A Cluster is a collection of particles.

   .. method:: split([axis], [random], [normal], [point])

      Splits the cluster into two clusters, where the first one is the original
      cluster and the new one is a new 'daughter' cluster.

      split is discussed in detail in :ref:`Mitosis and Events`


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

   .. method:: pressure()

      Computes the :ref:`Pressure Tensor` for the particles in this cluster. 

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

