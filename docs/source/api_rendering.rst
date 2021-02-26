Rendering and System Interaction
--------------------------------

.. module:: mechanica.system

The system module contains various function to interact with rendering engine
and host cpu.

.. function:: cpuinfo()

   Returns a dictionary that contains a variety of information about the
   current procssor, such as processor vendor and supported instruction set
   features. 

.. function:: gl_info()

   Returns a dictionary that contains OpenGL capabilities and supported
   extensions.

.. function:: egl_info()

   Gets a string of EGL info, only valid on Linux, we use EGL for Linux headless
   rendering.

.. function:: image_data()

   Gets the contents of the rendering frame buffer back as a JPEG image stream,
   a byte array of the packed JPEG image.

.. function:: context_has_current()

   Checks of the currently executing thread has a rendering context. 

.. function:: context_make_current()

   Makes the single Mechanica rendering context current on the current
   thread. Note, for multi-threaded rendering, the initialization thread needs
   to release the context via :func:`context_release`, then aquire it on the
   rendering thread via this function.

.. function:: context_release()

   Release the rendering context on the currently executing thread.

.. function:: camera_rotate(euler_angles)

   Rotate the current scene about the given Euler angles, where ``euler_angles``
   is a length-3 array.

   :param euler_angles: 



