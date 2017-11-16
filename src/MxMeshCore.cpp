/*
 * MxMeshCore.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include "MxMeshCore.h"
#include "MxTriangle.h"

std::set<VertexPtr> MxVertex::link() const {
    std::set<VertexPtr> lnk;

    for(TrianglePtr tri : _triangles) {
        for(VertexPtr v : tri->vertices) {
            if(v != this) {lnk.insert(v);}
        }
    }
    return lnk;
}

HRESULT MxVertex::removeTriangle(const TrianglePtr tri) {
    auto iter = std::find(_triangles.begin(), _triangles.end(), tri);
    if(iter != _triangles.end()) {
        _triangles.erase(iter);
        rebuildFacets();
        return S_OK;
    }
    return E_FAIL;
}

HRESULT MxVertex::appendTriangle(TrianglePtr tri) {
    if(!contains(_triangles, tri)) {
        _triangles.push_back(tri);
        rebuildFacets();
        return S_OK;
    }
    return E_FAIL;
}

void MxVertex::rebuildFacets() {
    _facets.clear();
    for(TrianglePtr tri : _triangles) {
        if(!contains(_facets, tri->facet)) {
            _facets.push_back(tri->facet);
        }
    }
}
