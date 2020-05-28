Forces
------


Forces are one of the fundamental processes in Mechanica that cause objects to
move. We provide a suite of pre-built forces, and users can create their own.

some stuff here


.. module::  mechanica.forces

The forces module collects all of the built-in forces that Mechanica
provides. This module contains a variety of functions that all generate a
force object.


.. function:: berendsen_thermostat(tau) 

   Creates a Berendsen thermostat

   :param: tau: time constant that determines how rapidly the thermostat effects
           the system. 
          
   The Berendsen thermostat effectively re-scales the velocities of an object in
   order to make the temperature of that family of objects match a specified
   temperature. 

   .. math::

      \frac{d \mathbf{p}_i}{dt} += \frac{\mathbf{p}_i}{\tau_T}
          \left(\frac{T_0}{T} - 1 \right)



      
