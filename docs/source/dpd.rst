The bulk motion or advection time evolution of a solvent tDPD bulk particle
:math:`i` obeys both conservation of momentum and mass (solute amount), and is
generally written as:

.. math::

   \frac{d^2\mathbf{r}_i}{dt^2} = \frac{d \mathbf{v}_i}{dt} = \sum_{i \neq j}
   \left( \mathbf{F}^C_{ij} + \mathbf{F}^D_{ij} + \mathbf{F}^R_{ij} \right)
   + \mathbf{F}^{ext}_i,
     
where  :math:`t`, :math:`\mathbf{r}_i`, :math:`\mathbf{v}_i`,
:math:`\mathbf{F}` are time, position velocity, and force vectors,
respectively, and :math:`\mathbf{F}_{ext}` is the external force on particle
:math:`i`. Forces :math:`\mathbf{F}^C_{ij}`, :math:`\mathbf{F}^D_{ij}` and
:math:`\mathbf{F}^R_{ij}` are the pairwise conservative, dissipative and random
forces respectively.

The conservative force represents the inertial forces in the fluid, and is
typically a Lennard-Jones 12-6 type potential. The dissipative, or friction
force :math:`\mathbf{F}^D` represents the dissipative forces, and the random
force :math:`\mathbf{F}^R` is a pair-wise random force between particles. Users
are of course free to choose any forces they like, but these are the most
commonly used DPD ones. 


The pairwise forces are commonly expressed as:

.. math::

   \begin{eqnarray}
     \mathbf{F}^C_{ij} &=& a_{ij}\left(1 - \frac{r_{ij}}{r_c}\right)\mathbf{e}_{ij}, \\
     \mathbf{F}^D_{ij} &=& -\gamma_{ij}\left(1 - \frac{r_{ij}}{r_c}\right)^{0.41}(\mathbf{e}_{ij} \cdot
     \mathbf{v}_{ij}) \mathbf{e}_{ij}, \\
     \mathbf{F}^R_{ij} &=& \sigma_{ij}\left(1 - \frac{r_{ij}}{r_c}\right)^{0.2} \xi_{ij}\Delta t^{-1/2}\mathbf{e}_{ij},
   \end{eqnarray}

Here, :math:`r_{ij} = |\mathbf{r}_{ij}|`, :math:`\mathbf{r}_{ij} =
\mathbf{r}_i - \mathbf{r}_j`, :math:`\mathbf{e}_{ij} = \mathbf{r}_{ij} /
r_{ij}`.  :math:`\mathbf{v}_{ij} = \mathbf{v}_i -
\mathbf{v}_j`.

All of these pairwise forces are conveniently  available in the ``forces`` package
as the :any:`forces.dpd_conservative`, :any:`forces.dpd_dissipative` and
:any:`forces.dpd_random` respectively. 




The parameters in the tDPD system are defined as

* :math:`\rho = 4.0`
* :math:`k_BT=1.0`
* :math:`a=75k_B T/ \rho`
* :math:`\gamma=4.5`
* :math:`\omega^2=2k_B T \gamma`
* :math:`r_c=r_{cc} = 1.58`
* :math:`\omega_C(r) = (1 - r/r_c)`
* :math:`\omega_D(r) = \omega^2_R(r) = (1 -r/r_c)^{0.41}` 
* :math:`\omega_{DC}(r) = (1 - r/r_{cc})^2`

and :math:`\kappa` ranges from 0 to 10.




