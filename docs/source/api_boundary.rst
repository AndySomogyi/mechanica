Boundary Conditions
-------------------

.. class:: BoundaryCondition

   .. attribute:: name

      Gets the name of the boundary condition, i.e. "top", "bottom", etc..

   .. attribute:: kind

      The 'kind' of the boundary condition, i.e. "PERIODIC", "FREESLIP", or
      "VELOCITY"

   .. attribute:: velocity

      gets / sets the velocity of the boundary, a length-3 vector. Note a
      "no-slip" boundary is just a "velocity" boundary with a zero velocity
      vector.

   .. attribute:: normal

      get the normal vector of the boundary, points *into* the simulation
      domain.

   .. attribute:: restore

      gets / sets the restoring percent. When objects hit this boundary, they
      get reflected back at `restore` percent, so if restore is 0.5, and object
      hitting the boundary at 3 length / time recoils with a velocity of 1.5
      lengths / time. 



.. class:: Boundaryconditions

   The BoundaryConditions class really just serves as a contianer for the six
   instances of the BoundaryCondition object:


   .. attribute:: left

   .. attribute:: right

   .. attribute:: front

   .. attribute:: back

   .. attribute:: bottom

   .. attribute:: top





Boundary Condition Constants
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: BOUNDARY_NONE

   no boundary conditions

.. py:data:: PERIODIC_X

   periodic in the x direction

.. py:data:: PERIODIC_Y

   periodic in the y direction

.. py:data:: PERIODIC_Z

   periodic in the z direction

.. py:data:: PERIODIC_FULL

   periodic in all directions

.. py:data:: PERIODIC_GHOST_X
.. py:data:: PERIODIC_GHOST_Y
.. py:data:: PERIODIC_GHOST_Z
.. py:data:: PERIODIC_GHOST_FULL

.. py:data:: FREESLIP_X

   free slip in the x direction

.. py:data:: FREESLIP_Y

   free slip in the y direction

.. py:data:: FREESLIP_Z

   free slip in the z direction

.. py:data:: FREESLIP_FULL

   free slip in all directions

.. _geometry_constants_label








