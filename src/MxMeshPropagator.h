/*
 * MxMeshPropagator.h
 *
 *  Created on: Jul 11, 2017
 *      Author: andy
 */

#ifndef SRC_MXMESHPROPAGATOR_H_
#define SRC_MXMESHPROPAGATOR_H_

/**
 * Defines and calculates the time evolution of a mesh. The MxMesh defines the
 * present state of a system, and the propagator acts on the mesh to step it
 * forward in time.
 *
 * We expect to have many different kinds of propagators, such as Newtonian
 * dynamics, friction dominated dynamics, etc...
 *
 * The propagator also computes the time evolution of the scalar fields that are
 * attached to each vertex/facet of the mesh.
 */
class MxMeshPropagator {
public:
    MxMeshPropagator();
    virtual ~MxMeshPropagator();
};

#endif /* SRC_MXMESHPROPAGATOR_H_ */
