Simulator
---------

.. class:: Simulator(object)

   The Simulator is the entry point to simulation, this is the very first object
   that needs to be initialized  before any other method can be called. All the
   methods of the Simulator are static, but the constructor needs to be called
   first to initialize everything.

   The Simulator manages all of the operating system interface, it manages
   window creation, end user input events, GPU access, threading, inter-process
   messaging and so forth. All 'physical' modeling concepts to in the
   :class:`Universe` object. 

   .. method:: Simulator.__init__(self, conf=None, )

      Initializes a simulation. All of the keyword arguments are the same as on the
      Config object. You can initialize the simulator via a config :class:`.Config`
      object , or via keyword arguments. The keywords have the same name as fields
      on the config. 

   .. staticmethod:: run()

      Starts the operating system messaging event loop for application. This is
      typically used from scripts, as console input will no longer work. The run
      method will continue to run until all of the windows are closed, or the
      ``quit`` method is called. By default, ``run`` will automatically start
      the universe time propagation.
      

   .. staticmethod:: irun()

      Runs the simulator in interactive mode, in that in interactive mode, the
      console is still active and users can continue to issue commands via the
      ipython console. By default, ``irun`` will automatically start the
      universe time propagation.
      

   .. staticmethod:: quit()

      Terminates the message loop. The message loop is automatically terminated
      when all the active windows close.

   .. staticmethod:: show()

      Shows any windows that were specified in the config. This works just like
      MatPlotLib's ``show`` method. The ``show`` method does not start the
      universe time propagation unlike ``run`` and ``irun``.

.. class:: Simulator.Config()

   An object that has all the arguments to the simulator, 



Event / Message Processing
--------------------------


void glfwPollEvents	(	void 		)	
This function processes only those events that are already in the event queue and then returns immediately. Processing events will cause the window and input callbacks associated with those events to be called.

On some platforms, a window move, resize or menu operation will cause event processing to block. This is due to how event processing is designed on those platforms. You can use the window refresh callback to redraw the contents of your window when necessary during such operations.
