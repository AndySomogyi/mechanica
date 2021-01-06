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



      .. staticmethod:: well(k, n, r0, [min], [max], [tol])

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


      .. figure:: square_well_plot.png
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

      .. staticmethod:: soft_sphere(kappa, epsilon, r0, eta, min, max, tol)

         Creates a soft sphere interaction potential. The soft sphere is a
         generalized Lennard Jones type potentail, but you can varry the
         exponents to create a softer interaction.

         :param kappa:
         :param epsilon:
         :param r0:
         :param eta:
         :param min:
         :param max:
         :param tol:


      .. staticmethod:: glj(e, [m], [n], [r0], [min], [max], [tol], [shifted])

         :param e: effective energy of the potential 
         :param m: order of potential, defaults to 3
         :param n: order of potential, defaults to 2*m
         :param r0: mimumum of the potential, defaults to 1
         :param min:  minimum distance, defaults to 0.05 * r0
         :param max:  max distance, defaults to 5 * r0
         :param tol:  tolerance, defaults to 0.01
         :param shifted: is this shifted potential, defaults to true 

         Generalized Lennard-Jones potential.

         .. math::

            V^{GLJ}_{m,n}(r) = \frac{\epsilon}{n-m} \left[ m \left( \frac{r_0}{r}
            \right)^n - n \left( \frac{r_0}{r} \right) ^ m \right]

         where :math:`r_e` is the effective radius, which is automatically
         computed as the sum of the interacting particle radii.

         .. figure:: glj.png
            :width: 500px
            :align: center
            :alt: alternate text
            :figclass: align-center

               

            The Generalized Lennard-Jones potential for different exponents
            :math:`(m, n)` with fixed :math:`n = 2m`.  As the exponents grow smaller,
            the potential flattens out and becomes softer, but as the exponents grow
            larger the potential becomes narrower and sharper, and approaches
            the hard sphere potential.

                  

      .. staticmethod:: overlapping_sphere(mu=1, [kc=1], [kh=0], [r0=0], [min=0.001], max=[10], [tol=0.001])


         :param mu: interaction strength, represents the potential energy peak
                    value.
         :param kc: decay strength of long range attraction. Larger values make
                    a shorter ranged function.  
         :param kh: Optionally add a harmonic long-range attraction, same as
                    :meth:`glj` function.
         :param r0: Optional harmonic rest length, only used if `kh` is
                    non-zero. 
         :param min: Minimum value potential is computed for. 
         :param max: Potential cutoff values.
         :param tol: Tolerance, defaults to 0.001.


         The `overlapping_sphere` function implements the `Overlapping Sphere`,
         from :cite:`Osborne:2017hk`. This is a soft sphere, from our
         testing, it appears *too soft*, probably better suited for 2D
         models. This potential appears to allow particles to collapse too
         closely, probably needs more paramater fiddling.

         .. note::
            From the equation below, we can see that there is a :math:`\log`
            term as the short range repulsion term. The logarithm is the radial
            Green's function for cylindrical (2D) geometry, however the Green's
            function for 3D is the :math:`1/r` function. This is possibly why
            Osborne has success in 2D but it's unclear if this was used in 3D
            geometries. 


         .. math::

            \mathbf{F}_{ij}= 
            \begin{cases}
              \mu_{ij} s_{ij}(t) \hat{\mathbf{r}}_{ij} \log 
                \left(
                1 + \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)}
                \right) ,& \text{if } ||\mathbf{r}_{ij}|| < s_{ij}(t) \\
              \mu_{ij}\left(||\mathbf{r}_{ij}|| - s_{ij}(t)\right) \hat{\mathbf{r}}_{ij} 
                \exp \left( 
                -k_c \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)}
                \right) ,&
                \text{if } s_{ij}(t) \leq ||\mathbf{r}_{ij}|| \leq r_{max} \\
              0,              & \text{otherwise} \\
            \end{cases}

         Osborne refers to :math:`\mu_{ij}` as a "spring constant", this
         controls the size of the force, and is the potential energy peak value.
         :math:`\hat{\mathbf{r}}_{ij}`  is the unit vector from particle
         :math:`i` center to particle :math:`j` center, :math:`k_C` is a
         parameter that defines decay of the attractive force. Larger values of
         :math:`k_C` result in a shaper peaked attraction, and thus a shorter
         ranged force. :math:`s_{ij}(t)` is the is the sum of the radii of the
         two particles.

         We can plot the overlapping sphere function to get an idea of it's
         behavior::

         >>> import mechanica as m
         >>> p = m.Potential.overlapping_sphere(mu=10, max=20)
         >>> p.plot(s=2, force=True, potential=True, ymin=-10, ymax=8)

      .. figure:: overlapping_sphere.png
         :width: 500px
         :align: center
         :alt: alternate text
         :figclass: align-center

         We can plot the overlapping sphere function like any other function.

         
   
 


      
