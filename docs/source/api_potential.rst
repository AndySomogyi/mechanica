Potentials
----------

A Potential object is a compiled interpolation of a given function. The
Universe applies potentials to particles to calculate the net force on them.

For performance reasons, we found that implementing potentials as
interpolations can be much faster than evaluating the function directly.

A potential can be treated just like any python callable object to evaluate it::

  >>> pot = m.Potential.lennard_jones_12_6(0.1, 5, 9.5075e-06 , 6.1545e-03, 1.0e-3 )
  >>> x = np.linspace(0.1,1,100)
  >>> y=[pot(j) for j in x]
  >>> plt.plot(x,y, 'r')

.. figure:: lj_figure.png
    :width: 500px
    :align: center
    :alt: alternate text
    :figclass: align-center

    Potential is a callable object, we can invoke it like any Python function. 


.. class:: Potential

   
   .. staticmethod:: lennard_jones_12_6(min, max, a, b, )

      Creates a Potential representing a 12-6 Lennard-Jones potential
 
      :param min: The smallest radius for which the potential will be constructed.
      :param max: The largest radius for which the potential will be constructed.
      :param A:   The first parameter of the Lennard-Jones potential.
      :param B:   The second parameter of the Lennard-Jones potential.
      :param tol: The tolerance to which the interpolation should match the exact
             potential., optional
 
      The Lennard Jones potential has the form:

      .. math::
         \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right)


   .. staticmethod:: lennard_jones_12_6_coulomb(min, max, a, b, tol )

      Creates a Potential representing the sum of a
       12-6 Lennard-Jones potential and a shifted Coulomb potential.
 
      :param min: The smallest radius for which the potential will be constructed.
      :param max: The largest radius for which the potential will be constructed.
      :param A: The first parameter of the Lennard-Jones potential.
      :param B: The second parameter of the Lennard-Jones potential.
      :param q: The charge scaling of the potential.
      :param tol: The tolerance to which the interpolation should match the exact
       potential. (optional)
 
      The 12-6 Lennard Jones - Coulomb potential has the form:

      .. math::
         \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right)
         + q \left(\frac{1}{r} - \frac{1}{max} \right)

   .. staticmethod:: ewald(min, max, q, kappa, tol)

      Creates a potential representing the real-space part of an Ewald 
      potential.
 
      :param min: The smallest radius for which the potential will be constructed.
      :param max: The largest radius for which the potential will be constructed.
      :param q: The charge scaling of the potential.
      :param kappa: The screening distance of the Ewald potential.
      :param tol: The tolerance to which the interpolation should match the exact
                  potential.

      The Ewald potential has the form:

      .. math::
 
         q \frac{\mbox{erfc}( \kappa r)}{r}



      .. staticmethod:: wall(k, n, r0, [min], [max], [tol])

         Creates a continuous square well potential. Usefull for binding a
         particle to a region.


         :param float k:   potential prefactor constant, should be decreased for
                           larger n.
         :param float n:   exponent of the potential, larger n makes a sharper
                           potential.
         :param float r0:  The extents of the potential, length units. Represents
                           the maximum extents that a two objects connected with
                           this potential should come appart. 
         :param float min: [optional] The smallest radius for which the potential
                           will be constructed, defaults to zero. 
         :param float max: [optional]  The largest radius for which the potential
                           will be constructed, defaults to r0. 
         :param float tol: [optional[ The tolerance to which the interpolation
                           should match the exact potential, defaults to 0.01 *
                           abs(min-max).  
 
      .. math::

         \frac{k}{\left(r_0 - r\right)^{n}}

      As with all potentials, we can create one, and plot it like so::

        >>> p = m.Potential.well(0.01, 2, 1)
        >>> x=n.arange(0, 1, 0.0001)
        >>> y = [p(xx) for xx in x]
        >>> plt.plot(x, y)
        >>> plt.title(r"Continuous Square Well Potential $\frac{0.01}{(1 - r)^{2}}$ \n",
        ...           fontsize=16, color='black')


      .. figure:: square_well.png
         :width: 500px
         :align: center
         :alt: alternate text
         :figclass: align-center

         A continuous square well potential.



      .. staticmethod:: harmonic_angle(k, theta0, [min], max, [tol])

         Creates a harmonic angle potential
 
         :param k: The energy of the angle.
         :param theta0: The minimum energy angle.
         :param min: The smallest angle for which the potential will be constructed.
         :param max: The largest angle for which the potential will be constructed.
         
         :param tol: The tolerance to which the interpolation should match the exact
                     potential.
 
         returns A newly-allocated potential representing the potential

         .. math::
            k(\theta-\theta_0)^2

         Note, for computational effeciency, this actually generates a function
         of r, where r is the cosine of the angle (calculated from the dot
         product of the two vectors. So, this actually evaluates internally,

         .. math::
            k(\arccos(r)-\theta_0)^2 
         

      .. staticmethod:: harmonic(k, r0, [min], [max], [tol])

         Creates a harmonic bond potential

         :param k: The energy of the bond.
         :param r0: The bond rest length
         :param min: [optional] The smallest radius for which the potential will
                     be constructed. Defaults to :math:`r_0 - r_0 / 2`. 

         :param max: [optional] The largest radius for which the potential will
                     be constructed. Defaults to :math:`r_0 + r_0 /2`.

         :param tol: [optional] The tolerance to which the interpolation should
                     match the exact potential. Defaults to :math:`0.01 \abs(max-min)`
 
         return A newly-allocated potential

         .. math::

            k (r-r_0)^2
 




   
 


      
