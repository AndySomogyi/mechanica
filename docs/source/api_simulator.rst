Simulator
---------

.. data:: version(str)

   The current Mechanica version, as a string.


.. class:: Simulator(object)

   The Simulator is the entry point to simulation, this is the very first object
   that needs to be initialized  before any other method can be called. All the
   methods of the Simulator are static, but the constructor needs to be called
   first to initialize everything.

   The Simulator manages all of the operating system interface, it manages
   window creation, end user input events, GPU access, threading, inter-process
   messaging and so forth. All 'physical' modeling concepts to in the
   :class:`Universe` object. 

   .. method:: Simulator.__init__(self, conf=None, **kwargs)

      The ``Simulator(...)`` construtor call is really just a synonum for the
      main :func:`init` function. We keep this call here as earlier versions
      used this as the init. 

        
        

   .. staticmethod:: run()

      alias to top level :func:`mechanica.run` function.

      

   .. staticmethod:: irun()

      alias to top level :func:`mechanica.irun` function.

      
      

   .. staticmethod:: close()

      alias to top level :func:`mechanica.close` function.


   .. staticmethod:: show()

      alias to top level :func:`mechanica.show` function.

      
.. class:: Simulator.Config()

   An object that has all the arguments to the simulator, 



Event / Message Processing
--------------------------


void glfwPollEvents	(	void 		)	
This function processes only those events that are already in the event queue and then returns immediately. Processing events will cause the window and input callbacks associated with those events to be called.

On some platforms, a window move, resize or menu operation will cause event processing to block. This is due to how event processing is designed on those platforms. You can use the window refresh callback to redraw the contents of your window when necessary during such operations.
