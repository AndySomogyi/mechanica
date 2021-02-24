Fluxes
------

.. module:: mechanica

.. function:: flux(a, b, species_name, k, [decay=0])

   The basic passive (Fickian) flux type. This flux implements a passive
   transport between a species located on a pair of nearby objects of type a
   and b. See :ref:`flux-label` for a detailed
   discussion.

   :param type a: object type a
   :param type b: object type b
   :param string species_name:  string, textual name of the species to perform a flux with
   :param number k: flux rate constant
   :param number decay: decay rate, optional, defaults to 0.


   A Fick flux of the species :math:`S` attached to object types
   :math:`A` and :math:`B` implements the reaction:

   .. math::

      \begin{eqnarray}
      a.S & \leftrightarrow a.S \; &; \; k \left(1 - \frac{r}{r_{cutoff}} \right)\left(a.S - b.S\right)     \\
      a.S & \rightarrow 0   \; &; \; \frac{d}{2} a.S \\
      b.S & \rightarrow 0   \; &; \; \frac{d}{2} b.S,
      \end{eqnarray}

   :math:`B` respectivly. :math:`S` is a chemical species located at each
   object instances. :math:`k` is the flux constant, :math:`r` is the
   distance between the two objects, :math:`r_{cutoff}` is the global cutoff
   distance, and :math:`d` is the optional decay term. 



.. function:: produce_flux(a, b, species_name, k, target,  [decay=0])

   An active flux that represents active transport, and is can be used to model
   such processes like membrane ion pumps. See :ref:`flux-label` for a detailed
   discussion. Unlike the :ref:`consume_flux`, the produce flux uses the
   concentation of only the source to determine the rate. 

   :param type a: object type a
   :param type b: object type b
   :param string species_name:  string, textual name of the species to perform a flux with
   :param number k: flux rate constant
   :param number target: target concentation of the :math:`b.S`
                                      species. 
   :param number decay: decay rate, optional, defaults to 0.


   The produce flux implements the reaction:

   .. math::
      \begin{eqnarray}
      a.S & \rightarrow b.S \; &; \;  k \left(r - \frac{r}{r_{cutoff}} \right)\left(a.S - a.S_{target} \right) \\
      a.S & \rightarrow 0   \; &; \;  \frac{d}{2} a.S \\
      b.S & \rightarrow 0   \; &; \;  \frac{d}{2} b.S
      \end{eqnarray}



.. function:: consume_flux(a, b, species_name, k, target_concentation,  [decay=0])

   An active flux that represents active transport, and is can be used to model
   such processes like membrane ion pumps. See :ref:`flux-label` for a detailed
   discussion. 

   :param type a: object type a
   :param type b: object type b
   :param string species_name:  string, textual name of the species to perform a flux with
   :param number k: flux rate constant
   :param number target: target concentation of the :math:`b.S`
                                      species. 
   :param number decay: decay rate, optional, defaults to 0.


   The consume flux implements the reaction:

   .. math::
      \begin{eqnarray}
      a.S & \rightarrow b.S \; &; \; k \left(1 - \frac{r}{r_{cutoff}}\right)\left(b.S - b.S_{target} \right)\left(a.S\right) \\
      a.S & \rightarrow 0   \; &; \; \frac{d}{2} a.S \\
      b.S & \rightarrow 0   \; &; \; \frac{d}{2} b.S
      \end{eqnarray}




   
