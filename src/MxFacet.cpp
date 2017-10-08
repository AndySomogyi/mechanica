/*
 * MxFacet.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <MxFacet.h>

MxFacet::MxFacet(MxFacetType* type, MeshPtr msh, const std::array<CellPtr, 2>& cells) :
    MxObject{type}, MxMeshNode{msh}, cells{cells} {
}

HRESULT MxFacet::appendChild(TrianglePtr tri) {
    if(contains(triangles, tri)) {
        return mx_error(E_FAIL, "already contains triangle");
    }

    if(tri->facet) {
        return mx_error(E_FAIL, "triangle belongs to another facet");
    }

    for(VertexPtr v : tri->vertices) {
        if(!contains(v->facets, this)) {
            v->facets.push_back(this);
        }
    }

    tri->facet = this;
    triangles.push_back(tri);
    return S_OK;
}

HRESULT MxFacet::removeChild(TrianglePtr tri) {
    if(tri->facet != this) {
        return mx_error(E_FAIL, "triangles does not belong to this facet");
    }
    std::remove(triangles.begin(), triangles.end(), tri);
    tri->facet = nullptr;
    return S_OK;
}

HRESULT MxFacet::positionsChanged() {
}
