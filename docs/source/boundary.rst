Boundary Conditions
===================


.. py:currentmodule:: mechanica

.. _boundary:


We can specify a variety of different boundary conditions via the `bc` argument
to the :any:`Simulator` constructor. We offer a range of boundary condition
options in the :ref:`Boundary Condition Constants` section.


We specify boundary conditions via the ``bc``  argument to the main
:func:`init` function call. Boundary conditions can be one of the simple kinds
if we use the numeric argument from the :ref:`Boundary Condition
Constants`, or can be a dictionary to use flexible boundary conditions.

For flexible boundadary conditions, we pass a dictionary to the ``bc``
:func:`init` argument like::

   m.init(bc={'x':'periodic', 'z':'no_slip', 'y' : 'periodic'})

The top-level keys in the ``bc`` dictionary can be either ``"x"``, ``"y"``, ``"z"``, or
``"left"``, ``"right"``, ``"top"``, ``"bottom"``, ``"front"``, or ``"back"``. If
we choose one of the axis directions, ``"x"``, ``"y"``, ``"z"``, then
boundary condition is symmetric, i.e. if we set ``"x"`` to some value, then both
``"left"`` and ``"right"`` get set to that value. Valid options for axis
symmetric boundaries are ``"periodic"``, ``"freeslip"``, ``"noslip"`` or
``"potential"``.






