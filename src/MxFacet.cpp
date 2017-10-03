/*
 * MxFacet.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <MxFacet.h>

MxFacet::MxFacet(MxFacetType* type, MeshPtr msh, const std::array<CellPtr, 2>& cells) :
	MxObject{type}, MxMeshNode{msh} {
}

HRESULT MxFacet::appendChild(TrianglePtr tri) {
}
