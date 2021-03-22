Mechanica API Reference
#######################

.. py:currentmodule:: mechanica

This is the API Reference page for the module: :mod:`mechanica`


.. module:: mechanica
   :platform: OSX, Linux, Windows 
   :synopsis: Simulate and Analyze Active Mater

.. moduleauthor:: Andy Somogyi <andy.somogyi@gmail.com>


.. function:: init(...)

   Initializes a simulation. All of the keyword arguments are the same as on the
   Config object. You can initialize the simulator via a config :class:`.Config`
   object , or via keyword arguments. The keywords have the same name as fields
   on the config.


        

.. function:: run()

   Starts the operating system messaging event loop for application. This is
   typically used from scripts, as console input will no longer work. The run
   method will continue to run until all of the windows are closed, or the
   ``quit`` method is called. By default, ``run`` will automatically start
   the universe time propagation.
      

.. function:: irun()

   Runs the simulator in interactive mode, in that in interactive mode, the
   console is still active and users can continue to issue commands via the
   ipython console. By default, ``irun`` will automatically start the
   universe time propagation.
      

.. function:: close()

   Closes the main window, but the application / simulation will continue to
   run. 


.. function:: show()

   Shows any windows that were specified in the config. This works just like
   MatPlotLib's ``show`` method. The ``show`` method does not start the
   universe time propagation unlike ``run`` and ``irun``.


   
   
   :Keyword Arguments:
      * *dim* (``vector3 like``) --
        [x,y,z] dimensions of the universe, defaults to [10., 10., 10.]
        
      * *bc* (``number or dictionary``) --
        Boundary conditions, use one of the constants in the 
        :ref:`Boundary Condition Constants` section. Defaults to
        :any:`PERIODIC_FULL`. For a detailed discussion of boundary
        condtion options, see :any:`boundary`
             
      * *cutoff* (``number``) --
        Cutoff distance for long range forces, try to keep small for best
        performance, defaults to 1
        
      * *cells* (``vector3 like``) --
        [x,y,z] dimensions of spatial cells, how we partition space. Defaults to [4,4,4]
        
      * *threads* (``number``) --
        number of compute threads, defaults to 4
        
      * *integrator* (``number``) --
        kind of integrator to use, defaults to :any:`FORWARD_EULER`
        
      * *dt* (``number``) --
        time step (:math:`\delta t`). If you encounter numerical instability,
        reduce time step, defaults to 0.01
        
        
   


.. include:: api_simulator.rst

.. include:: api_universe.rst

.. include:: api_boundary.rst

.. include:: api_constants.rst
   
.. include:: api_particle.rst

.. include:: api_particlelist.rst

.. include:: api_potential.rst

.. include:: api_forces.rst

.. include:: api_species.rst

.. include:: api_flux.rst

.. include:: api_bonded.rst

.. include:: api_events.rst

.. include:: api_style.rst

.. include:: api_logger.rst

.. include:: api_rendering.rst

