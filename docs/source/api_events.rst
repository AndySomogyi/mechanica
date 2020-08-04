Events
------

Events are the principle way users can attach connect thier own functions, and
built in methods event triggers. 


.. module::  mechanica


.. function:: on_time(method, period, [start], [end], [distribution]) 

   Binds a function or method to a elapsed time interval.

   on_time must be called with all keyword arguments, since there are a large
   number of options this method accepts.

   :param method: A Python function or method that will get called. Signature
           of the method should be ``method(time)``, in that it gets a single
           current time argument.

   :param float period: Time interval that the event should get fired.

   :param float start: [optional] Start time for the event.

   :param float end: [optional] end time for the event.

   :param str distribution: [optional] String that identifies the statistical distribution for
                        event times. Only supported distibution currently is
                        "exponential". If there is no `distibution` argument,
                        the event is called deterministically. 

   
