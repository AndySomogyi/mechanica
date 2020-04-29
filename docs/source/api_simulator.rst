Simulator
---------

The Simulator is the top level interface to all of the Mechanica functionality




.. class:: Simulator()

   Does stuff, makes of of these


.. method:: Simulator.doStuff()

   does some more stuff



Event / Message Processing
--------------------------


void glfwPollEvents	(	void 		)	
This function processes only those events that are already in the event queue and then returns immediately. Processing events will cause the window and input callbacks associated with those events to be called.

On some platforms, a window move, resize or menu operation will cause event processing to block. This is due to how event processing is designed on those platforms. You can use the window refresh callback to redraw the contents of your window when necessary during such operations.
