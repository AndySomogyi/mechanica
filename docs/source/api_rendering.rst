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




.. function:: camera_move_to([eye], [center], [up])

   Moves the camera to the specified 'eye' location, where it looks at the
   'center' with an up vector of 'up'. All of the parameters are optional, if
   they are not give, we use the initial default values.

   :param eye: ([x, y, z])  New camera location, where we re-position the eye of
                         the viewer.

   :param center: ([x, y, z]) Location that the camera will look at, the what we
                            want as the center of the view.

   :param up: ([x, y, z]) Unit up vector, defines the up direction of the
                        camera. 


.. function:: camera_reset()

   Resets the camera to the initial position. 


.. function:: camera_rotate_mouse(mouse_pos)

   Rotates the camera according to the current mouse position. Need to call
   :func:`camera_init_mouse()` to initialize a mouse movement.

   :param mouse_pos: ([x, y]) current mouse position on the view window or
                            image. 


.. function:: camera_translate_mouse(mouse_pos)

   Translates the camera according to the current mouse position. You need to
   call :func:`camera_init_mouse()` to initialize a mouse movment, i.e. set the
   starting mouse position.

   :param  mouse_pos: ([x, y]) current mouse position on the view window or image. 


.. function:: camera_init_mouse(mouse_pos)

   Initialize a mouse movment operation, this tells the simulator that a mouse
   click was performed at the given coordinates, and subsequent mouse motion
   will refer to this starting position.

   :param  mouse_pos: ([x, y]) current mouse position on the view window or image. 


.. function:: camera_translate_by(delta)

   Translates the camera in the plane perpendicular to the view orientation,
   moves the camera a given delta x, delta y distance in that plane.

   :param  delta: ([delta_x, delta_y]) a vector that indicates how much to
                                    translate the camera. 


.. function:: camera_zoom_by(delta)

   Zooms the camera in and out by a specified amount.

   :param delta: number that indicates how much to increment zoom distance. 


.. function:: camera_zoom_to(distance)

   Zooms the camera to the given distance.

   :param distance: distance to the universe center for the camera. 


.. function:: camera_rotate_to_axis(axis, distance)

   Rotates the camera to one of the principal axis, at a given zoom distance.

   :param  axis: ([x, y, z]) unit vector that defines the axis to move to.
   :param distance: how far away the camera will be. 


.. function:: camera_rotate_to_euler_angle(angles)

   Rotate the camera to the given orientiation defined by three Euler angles.

   :param  angles: ([alpha, beta, gamma]) Euler angles of rotation about the X, Y,
                                       and Z axis. 


.. function:: camera_rotate_by_euler_angle(angles)

   Incremetns the camera rotation by given orientiation defined by three Euler angles.

   :param  angles: ([alpha, beta, gamma]) Euler angles of rotation about the X, Y,
                                       and Z axis. 



.. function:: view_reshape(window_size)

   Notify the simulator that the window or image size was changed.

   :param  window_size: ([x, y]) new window size. 





