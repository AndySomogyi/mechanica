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
        rebuildCells();
        return S_OK;
    }
    return E_FAIL;
}

HRESULT MxVertex::appendTriangle(TrianglePtr tri) {
    if(!contains(_triangles, tri)) {
        _triangles.push_back(tri);
        rebuildCells();
        return S_OK;
    }
    return E_FAIL;
}

void MxVertex::rebuildCells() {
    _cells.clear();
    for(TrianglePtr tri : _triangles) {
        if(!contains(_cells, tri->cells[0])) {
            _cells.push_back(tri->cells[0]);
        }
        if(!contains(_cells, tri->cells[1])) {
            _cells.push_back(tri->cells[1]);
        }
    }
}
