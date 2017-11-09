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


/**
 * All of the triangles in a facet must have the correct winding, and the normal,
 * calculated by the vertex winding must point in the correct outward direction,
 * with respect to the cell at the cells[0] index.
 */
struct MxFacet : MxObject, MxMeshNode {

	MxFacet (MxFacetType *type, MeshPtr msh, const std::array<CellPtr, 2> &cells);

	/**
	 * Append a triangle to this facet. Examines the neighbors of the triangle, if the triangle
	 * has belongs to any facets that are not neighbors of this facet, than those facets
	 * are added to this facets list of neighbors.
	 */
	HRESULT appendChild(TrianglePtr tri);

	/**
	 * Removes the triangle from the list of triangles. Removes this facet from the
	 * triangle's list of facets.
	 */
	HRESULT removeChild(TrianglePtr tri);

    /**
     * Inform the facet that the vertex positions have changed. Causes the
     * cell to recalculate area and volume, also inform all contained objects.
     */
    HRESULT positionsChanged();

    /**
     * Need to associate this triangle with the cells on both sides. Trans-cell flux
     * is very frequently calculated, so optimize structure layout for both
     * trans-cell and trans-partial-triangle fluxes.
     */
    std::array<CellPtr, 2> cells;

    std::vector<TrianglePtr> triangles;

    std::vector<FacetPtr> neighbors;

    float area = 0;
};


#endif /* SRC_MXFACET_H_ */
