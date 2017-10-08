/*
 * ForceAccumulator.h
 *
 *  Created on: Oct 4, 2017
 *      Author: andy
 */

#ifndef SRC_FORCEACCUMULATOR_H_
#define SRC_FORCEACCUMULATOR_H_

#include "mechanica_private.h"
#include "MxMesh.h"

/**
 * Force Calculations
 *
 * One of the most expensive aspects of most physically based simulations such as
 * n-body (molecular dynamics, astrophysics), finite element, etc. is force calculations.
 *
 * In MD, force calculation, through efficient computationally and numeric implementation
 * is a challenge, is conceptually straightforward. (stuff about bonded and non-bonded
 * interactions) efficient numeric implementation via pipelining, cache coherence,
 * apply mathematical techniques such as operator splitting.
 *
 * The range and type of forces in active visco-elastic material is significantly
 * more rich and complex.
 *
 * Initial version will have three basic types of forces:
 * * Volume preservation -- acts perpendicular to surface
 * * Individual cell area preservation -- acts parallel to surface
 * * Shared contact area -- acts parallel to surface.
 *
 * The contact area forces represent surface and interfacial tension.
 *
 * The force calculations rely on numerous attributes, or 'bulk' quantities
 * of the objects that contain the triangles. It would of course be inefficient
 * to query and recalculate these quantities for each force calculation. Similarly
 * it is not quite architecturally pure for another component such as the propagator
 * to update these quantities, but the propagator is the component that actually
 * moves the vertices of each component. So, we have the propagator inform
 * each higher level component (cells) that it's vertices have moved, hence
 * it is time to re-calculate these bulk properties.
 *
 *
 *
 */


class ForceAccumulator;



typedef HRESULT (*MxCalcForce)(ForceAccumulator *, MxVertex **, uint32_t);


struct ForceAccumulatorType : MxType {
    MxCalcForce calculateForce;

};

class ForceAccumulator : MxObject {
public:

    HRESULT calculateForce(TrianglePtr* triangles, uint32_t len);

    //{
    //    return ((ForceAccumulatorType*)ob_type)->calculateForce(this, vertices, len);
    //}

private:

    //
    HRESULT volumeForce(TrianglePtr);

    //HRESULT
};

#endif /* SRC_FORCEACCUMULATOR_H_ */
