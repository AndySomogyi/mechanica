/*
 * MxPropagator.h
 *
 *  Created on: Apr 7, 2017
 *      Author: andy
 */

#ifndef SRC_MXPROPAGATOR_H_
#define SRC_MXPROPAGATOR_H_

#include "mechanica_private.h"

/**
 * Carry out time evolution of a model.
 *
 *
 * The model needs to compute a number of things that the propagator needs
 * to use to perform time evolution.
 *
 * Model must calculate forces acting on each elements, vertices, faces and volumes.
 * Ultimately, what move are the actual vertices, need to map forces acting on
 * high-dimensional elements onto the vertices. A number of different kinds of
 * forces exist in a model, these could be volume forces, surface forces, external
 * forces, etc.
 *
 * When we look at most MD packages, these perform bonded calculation on atom groups,
 * which is likely the most efficient approach. Here, atom groups depend on the kind of
 * force interaction, for example lets look at a two-body harmonic force. This force exists
 * between a pair of atom, and points either towards or directly away from the center of
 * geometry of this pair of atoms. The force is identical in magnitude, but opposite in
 * direction on each atom in the pair. Hence, the magnitude of the force calculation is
 * computed only once, but the the the force is added to each atom in the pair at the
 * same time (in opposite direction of course). All other kinds of bonded calculation are
 * performed the same way, i.e. a force can be a 1, 2, 3, 4, (nearly always less than 10),
 * and the force is applied to each atom in the group.
 *
 * Multiple different kinds of forces act on each atom, and each atom acts as an 'accumulator'
 * for the forces. Once all the force calculation are complete, the time evolution propagator
 * can then determine the time evolution based on the total force on each atom.
 *
 * Hence, one of the fundamental tasks of a model is to compute the total force that acts
 * on each vertex in a model.
 *
 * (1) forall vertices i
 * (2)     initialize xi = x0i ,vi = v0i ,wi = 1/mi
 * (3) endfor
 * (4) loop
 * (5)     forall vertices i do vi ← vi + ∆t * wi * force (xi)
 * (6)     dampVelocities(v1 , . . . , vN )
 * (7)     forallverticesidopi←xi+∆tvi
 * (8)     forall vertices i do generateCollisionConstraints(xi → pi)
 * (9)     loop solverIterations times
 * (10)        projectConstraints(C1,...,CM+Mcoll ,p1,...,pN)
 * (11)    endloop
 * (12)    forall vertices i
 * (13)        vi ←(pi−xi)/∆t
 * (14)        xi ← pi
 * (15)    endfor
 * (16)    velocityUpdate(v1 , . . . , vN )
 * (17) endloop
 *
 * There are a number of choices of how to compute the total force on each vertex.
 * The model can have many different kinds of forces. In MD, typically the MD engine
 * iterates over the bonded forces, and applies the force interaction to the atoms
 * specified in the force. One approach would be to simply let the model figure out
 * how to best calculate all force interactions, and all the propagator needs to know
 * is to call the model to carry out this task. Another approach would be for the
 * propagator to query the model for all of it's forces, iterate over them, and apply
 * them.
 *
 * Items to be aware of is that many different forces act on the same vertex, and
 * in a multi-threaded env, we can't iterate over the forces concurrently, we have to
 * iterate over the atoms.
 *
 * Should we accumulate force at the partial face level? Has the advantage that each
 * partial triangle belongs to exactly one cell, so we can concurrently iterate over cells
 * and compute the net force here. We know that the number of cells will nearly always be
 * Significantly larger than the number of processors, even in a GPU system.
 *
 * What is the best way to accumulate force. A single force vector per vertex is fine
 * a single threaded env, challenge is how to calculate the accumulation of forces when
 * multiple threads are involved and maintain efficiency. One option is to attach a set of
 * force vectors to each partial face. This approach doubles the force storage as each
 * vertex is attached to two faces. Does not completely solve the concurrency issue as
 * we could have external forces. Instead of three force vectors per triangle, could
 * attach force scalars that act towards and away from triangle center, and in the
 * triangle normal direction. This approach is less general, and makes adding new
 * forces types more difficult. Also does not completely solve the concurrency issue.
 *
 * Another options is to allocate a vector of force vectors per thread. This approach
 * is n-way redundant as the number of threads. Does make partitioning and load balancing
 * Significantly simpler as we do not have to perform complex graph partitioning. Can
 * also be readily adapted to GPUs. Combining total force per vertex is also quite simple
 * as all we have to do is uniformly partition the total system size by the number of threads,
 * and have each thread add up the total force contribution for that block.
 *
 * Traditional MD commonly partitions a system of N interacting atoms into regions of space
 * with each thread (or process) performs the total force and time calculation for all of the
 * atoms in this block. Requires introducing ghost atoms, these are a mirror copy of an atom
 * in a neighboring region.
 *
 * We can adopt a similar strategy with the observation that each vertex can belong to at
 * most two cells. What we can do is iterate over every face in each cell, and add that
 * cell's force contribution to each vertex. This approach requires that we have two
 * force vectors per vertex. We also store the partial triangle's id in the vertex, so
 * each vertex knows directly what partial faces it belongs to. With the PF id in each vertex,
 * we can uniquely identify which force vector in the pair of force vectors to accumulate
 * force to. Each cell with be processed by a single thread. External, non-bonded forces
 * will be very rarely used in these kinds of simulations, so in this case, we can simply
 * have external forces accumulate to say the zero'th force vector. This approach solves the
 * concurrency issue, and does not involve complex graph partitioning and ghost particles.
 *
 * The total cost is the addition of a force vector (12 bytes) per vertex. We also need to
 * identify which force vector the cell should add force to. One option would be to add a
 * either a bitfield, or a uchar per partial face. A more general approach is to add the
 * partial face identifier to each vertex. This enables us to traverse the graph in all
 * directions, and to calculate trans-membrane fluxes.
 *
 */
struct MxPropagator : MxObject {
};

HRESULT MxPropagator_init(PyObject *m);

#endif /* SRC_MXPROPAGATOR_H_ */
