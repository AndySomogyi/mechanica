Forces
------


Forces are one of the fundamental processes in Mechanica that cause objects to
move. We provide a suite of pre-built forces, and users can create their own.


.. module::  mechanica.forces

The forces module collects all of the built-in forces that Mechanica
provides. This module contains a variety of functions that all generate a
force object.


.. function:: berendsen_thermostat(tau) 

   Creates a Berendsen thermostat

   :param: tau: time constant that determines how rapidly the thermostat effects
           the system.

   The thermostat picks up the target temperature :math:`T_0` from the object
   that it gets bound to. For example, if we bind a temperature to a particle
   type, then it uses the 
          
   The Berendsen thermostat effectively re-scales the velocities of an object in
   order to make the temperature of that family of objects match a specified
   temperature.

   The Berendsen thermostat force :math:`\mathbf{F}_{temp}` has a function form of:

   .. math::

      \mathbf{F}_{temp} = \frac{\mathbf{p}_i}{\tau_T}
          \left(\frac{T_0}{T} - 1 \right),

   where :math:`T` is the measured temperature of a family of
   particles, :math:`T_0` is the control temperature, and
   :math:`\tau_T` is the coupling constant. The coupling constant is a measure
   of the time scale on which the thermostat operates, and has units of
   time. Smaller values of :math:`\tau_T` result in a faster acting thermostat,
   and larger values result in a slower acting thermostat.  


..class:: ConstantForce(value, period=None)

   A Constant Force that acts on objects, is a way to specify things like
   gravity, external pressure, etc...

   :param value: value can be either a callable that returns a length-3
                 vector, or a length-3 vector. If value is a vector, than we
                 simply use that vector value as the constant force. If value
                 is callable, then the runtime calls that callable every
                 period to get a new value. If it's a callable, then it must
                 return somethign convertible to a length-3 array. 
                 
   :param period: If value is a callable, then period defines how frequenly
                  the runtime should call that to obtain a new force value. 
                     
      
  
