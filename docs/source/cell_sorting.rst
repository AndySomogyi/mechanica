
Cell Sorting
============

In this section, we use cell sorting as a motivating example to develop a
mechanica model. We first introduce the biological phenomena, then we start our
analysis by first identifying what are the key objects and processes in the
biological descrioption, and how we represent these physical concepts with
Mechanica objects and processes.

Cell sorting between biological cells of different types is one of the basic
mechanisms creating tissue domains during develoment and wound healing and in
maintaining domains in homeostasis.  Cells of two different types, when
dissociated and randomly mixed can spontaneously sort to reform ordered
tissues. Cell sorting can generate regular patterns such as checkerboards,
engulfment of one cell type by another, or other more complex patterns
:cite:`Steinberg:1970bq`.


.. figure:: cell_sorting.jpg
    :width: 70 %
    :align: center

    A combination of cells from 3-germ layers are dissociiated and re-organize, 
    each cell type sorts out into its own region :cite:`Townes:1955ft`.



Both complete and partial cell sorting (where clusters of one cell type are trapped or contained inside a
closed envelope of another cell type) have been observed experimentally in vitro
in embryonic cells. Cell sorting does not involve cell
division nor differentiation but only spatial rearrangement of cell
positions. 



In a classic in vitro cell
sorting experiment to determine relative cell adhesivities in embryonic tissues,
mesenchymal cells of different types are dissociated, then randomly mixed and
reaggregated. Their motility and differential adhesivities then lead them to
rearrange to reestablish coherent homogenous domains with the most cohesive cell
type surrounded by the less-cohesive cell types :cite:`Armstrong:1972ep`
:cite:`Armstrong:1984tc`.


Cell-sorting behavior of cell aggregates is similar to liquid surface tension,
in the spontaneous separation of immiscible liquids (water vs. oil). Adhesive
forces between mixed cells play a similar role in cell sorting that
intermolecular attractive (cohesive) forces play in liquid surface tension. In
cell sorting, the cells with the strongest highest adhesivities will be sorted
to the center, while the less cohesive ones will remain outside.


Modeling and Abstraction
------------------------

To develop a computational model of our biological system, we must first identify the key objects and processes of our physical system. If we look at the left side of the following figure, we can see a sheet of biogical cells. From the previous description of the cell sorting, we also know that cells move about. We know that epethelial sheets are essentially a sheet of cells that form a surface. Here we can identify our first biogical object, a cell. From the previous discussion, we know that there are more than one type of cell, so lets call our two cell types, cell type A and cell type B. 


We can thus approximate an epethelial sheet as a two dimensional curved surface, and


.. _microscope-sheet-fig:

.. figure:: microscope-sheet.jpg
    :width: 70 %
    :align: center

    On the left is a confocal microscope image of a developing Drosophila wing,
    where the outlines of the cells have been highlighte with a florescent protein,
    which binds to E-cadherin, a surface proteion involved in cell adhesion. We
    can represent this sheet of biological cells with a set of polygons constrained
    to a two dimensional surface. Taken from :cite:`Fletcher:2014hub`

