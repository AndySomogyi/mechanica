Formalizing Physical Knowledge
******************************


The first step in formalizing knowledge is writing it down in such a way that it
has semantic meaning for both humans and computers. In order to specify
biological phenomena, we need a modeling description that corresponding closely
to natural phenomena.

Let's start by looking at the folliwing diagram of a single biological
cell. This cell could be moving on a surface, or inside some tissue. 

.. figure:: cell_1.jpg
    :width: 70 %
    :align: center

    A biological cell is a complex dynamic entity that participates in many
    active processes.

We can imedietly idenfity a number of *things*. Things such as cell nuclie, cell
membrane, cytoplasm, actin fibers, lamellpoidium, filopodium, etc.. We call
these things **objects**. Objects represent the “things” such as molecules, proteins,
cells, fluids, or materials. An object is defined as any instantiable physical
or logical entity that has certain state-full or structural
properties.

We also understand that for any living or active things, these objects typically
not static. Cells can move around, active fibers contract, cells can excert
forces on it's neighboring cells and enviormnet, cells 



Mechanica represents physical concepts using two key concepts: **objects** and
**processes**.

Processes represent anything that causes a change in state. A
process may, for example, represent a transformation in which sub-strates are
consumed and products are produced, or it may alter the state of an object,
either continuously or discretely. Unlike traditional programming languages,
processes in Mechanica can operate concurrently and continuously. This paradigm
of representing the world in terms of objects and processes forms the basis of
the Cellular Behavior Ontology (CBO) :cite:`Sluka:2014wz`, and also the Object
Process Methodology (OPM) :cite:`Dori:0pbBYLeH`.


Objects
=======

   	
**Objects** are the nouns, the actual things being described. Objects are such
things as molecules, cells, membranes, ion channels, the extra-cellular matrix,
fluids, etc. Objects have quantifiable characteristics such as location, amount,
concentration, mass and volume. Objects define a state; they are comparable to
standard data structures in conventional programming languages. Objects may
inherit and extend other objects, or may contain other objects. Objects are
grouped into two categories: continuous and discrete. Continous objects describe
things such as continuously valued chemical or molecular fields which have
position-dependent concentrations. Chemical fields may be 3D in that they are
bounded by a surface or 2D in that they exist on a surface. For reasons of
numerical efficiency, users may specify fields as spatially uniform.


Processes
=========
**Processes** are the verbs. Processes may create and destroy objects, alter the
state of objects, transform objects, or consume and produce sets of objects. As
in nature, multiple Mechannica processes act concurrently on objects and may act at
different rates, but may only be active under certain circumstances. Processes
may be continuously active, or may be explicitly triggered  by specific
conditions or invoked directly by some other process. Processes may also be
combined or aggregated, such that a process may be hierarchically composed of
child processes, and child processes may be ordered either concurrently or
sequentially. Processes fall into two categories: continuous and discrete.



A set of conditions may be attached to any process definition. Conditions are
specified in a \texttt{when} clause, much as they are in Modelica. Conditions
operate differently for discrete processes than they do for continuous
processes. Discrete processes are very similar to SBML events, in that a
discrete process is triggered only when its condition expression transitions
from false to true. The discrete process will also trigger at any future time
when the condition expression makes this transition. Conditions on a continous
process determine when the process should be active.


Discrete processes are  similar to functions in functional languages in that
they are a sequence of statements. The runtime triggers discrete processes when
their condition is met. Discrete processes can consume, create or modify
discrete objects. For example, users may define a discrete creation process
which creates new cell instances in a spatial region, or a discrete deletion
process which deletes a cell instance in response to some condition. The runtime
manages a pool of threads which execute triggered processes. The runtime also
continuously monitors each discrete process condition expression, and when the
condition evaluates to true, the runtime places the triggered process into a
priority queue and the next available thread executes the triggered processes.


Continuous processes ($\kappa$-processes) operate on continuous valued objects, and can be thought of as a generalization of the concept of chemical reactions. $\kappa$-processes consume reactants and produce products. Continuous processes must define a rate function which defines how fast the transformation (reaction) is occurring. The arguments to a $\kappa$-process must be labeled as either a reactant or a modifier, and a $\kappa$-process yields a set of zero or more products. An unlimited number of continuous $\kappa$-processes can act on an object instance, and the rate of change of this object instance is defined as the stoichiometric sum of all the currently active transformation processes that are consuming or producing this object. $\kappa$-processes may be used to define chemical reactions in a spatial compartment (such as cells) or membrane transport. 

Rate processes ($\rho$-processes) use a rate function to define the rate of change of a set of arguments. Only one $\rho$-process may be active on an object at a time; $\rho$-processes and $\kappa$-processes are mutually exclusive. 

Force processes ($\phi$-processes) provide a way to describe spatial change, such as motion, deformation, adhesion or response to  external forces. $\phi$-processes are similar to force functions in molecular dynamics. A $\phi$-process can be defined to act on one or between two spatial objects, hence a $\phi$-process may have one or two arguments, and both of them must be spatial object subtypes. $\phi$-processes return the force that acts on its arguments. Any motion processes (adhesion, taxis, deformation) can be specified via a suitable force process. For example, when an adhesion process is active between a surfaces of a pair of cells, the adhesion process applies a force between the cell surfaces at the locations where the surfaces are in contact. This adhesive force acts to keep the cells in contact and resists surface separation. 

The language runtime automatically applies the force functions to spatial objects and calculates the net force acting on each spatial object. The runtime then calculates the time evolution of each spatial object, typically as $\mathbf{v} \propto \mathbf{F}/m$, where velocity is proportional to the net force acting on each spatial object. 

Types
=====
**Types** serve to classify variable instances into categories. Every *thing* in
Mechanica (as well as Python and most programming languages)
has a well defined type. The type of a variable determines the kind of data that may be stored in that variable. The type of an object defines what operations are valid on instances of that object type, i.e., we can sum two numeric types, but adding a numeric type to a string is ill-defined. Most programming languages do not have a concept related to the biological notion of a phenotype. A phenotype in biology is a \emph{metric}, an observable categorization that determines a cell's type. A phenotype is defined by a set of rules or conditions such that when these conditions are met, we say that a cell is of such type.

The CCOPM extends the basic concept of dynamic or static types with a rule-based
type, which is related to the concept of typestate oriented programming :cite:`Strom:1986ht`. Here, the type may be defined via a set of rules, and when all of those rules are met, we say that a variable instance is a certain type. This notion is important because biological cells frequently undergo phenotypic change, that is, they change type over time. Processes are defined to operate on a specific type, and the runtime automatically applies these processes to all instances of the specified type. Here we can create a type definition based on a set of conditions; when all of these conditions are met, the type expression becomes true, and the processes corresponding to that type definition now automatically become active on all object instances for which the type evaluates to true. 

\textbf{Scope Resolution} determines how symbol names resolve to a
value. Programming languages typically have either static or dynamic scoping :cite:`grune2012modern`. Component composition in agent-based tissue simulations poses challenges that are not commonly encountered in traditional programming languages. Here, variables may also carry a spatial extent, for example, a chemical concentration exists over a region of space. So, whenever a chemical of a certain name is read or written, the value depends on where in space the read/write operation is occurring. Hence, the scope resolution in a spatial environment is related to the underlying spatial configuration. Furthermore, multiple different chemical networks may be placed inside of a cell or other spatial region. If this region is defined as a well-stirred compartment, all of the networks operate in the same space. Hence, any chemical species that these networks operate on must be connected to the same species in all of the other networks within that space. Additionally, chemical species may transfer across physical boundaries (e.g., cell membranes) that exist between distinct spatial regions. 

In order to account for the spatial nature of objects, we introduce a new scope
resolution rule which we call \emph{spatial scoping}. Spatial scoping extends
the traditional dynamic scoping with  environmental acquisition :cite:`Gil:1996wl`. Environmental acquisition was originally used in graphical user interface design to enable hierarchical composition of user interface widgets. In spatial scoping, scoping blocks correspond to a spatial region. In dynamic scoping, non-local symbols resolve to the scoping block where the function was called. In spatial scoping, non-local symbols resolve to the spatial region where the function is evaluated. This concept applies uniformly to spatially extended components such as chemical fields, as well as objects with well-defined boundaries such as cells. For example, a chemical network could exist inside of a cell. Here, each instance of that cell type will likely have different chemical values. The values that the chemical network processes read resolves to the local values found in each cell. Similarly, say we add a transport process to model an ion channel to a cell's surface, and this transport process's rate function defines a symbol corresponding to a chemical field. Even as the cell moves, the symbol always resolves to the value of the chemical field that corresponds to the cell surface location. 


