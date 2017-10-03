/*
 * MxFacet.h
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#ifndef SRC_MXFACET_H_
#define SRC_MXFACET_H_

#include "MxMeshCore.h"
#include "MxTriangle.h"

struct MxFacetType : MxType {

};

struct MxFacet : MxObject {

	MxFacet (MxFacetType *type, const std::array<CellPtr, 2> &cells);

    /**
     * Need to associate this triangle with the cells on both sides. Trans-cell flux
     * is very frequently calculated, so optimize structure layout for both
     * trans-cell and trans-partial-triangle fluxes.
     */
    std::array<CellPtr, 2> cells;

    std::vector<TrianglePtr> triangles;

    std::vector<FacetPtr> neighbors;
};


#endif /* SRC_MXFACET_H_ */
