/*
 * ConeSplit.cpp
 *
 *  Created on: Nov 30, 2017
 *      Author: andy
 */

#include "ConeSplit.h"
#include "MxMesh.h"

#include <set>


/**
 * List of triangles that make up a fan centered at vert. The fan
 * has one cell (cell) on one side, and two or more cells on the other
 * side.
 */
struct Cone {
    Cone(CVertexPtr vert, TrianglePtr tri, CellPtr cell) : centerCell{cell} {
        TrianglePtr first = tri;
        TrianglePtr prev = nullptr;
        do {
            triangles.push_back(tri);
            cells.push_back((tri->cells[0] == cell) ? tri->cells[1] : tri->cells[0]);
            TrianglePtr next = tri->nextTriangleInFan(vert, cell, prev);
            prev = tri;
            tri = next;
        } while(tri && tri != first);
        assert(triangles.size() > 2);
    }
    std::vector<TrianglePtr> triangles;
    std::vector<CellPtr> cells;
    CellPtr centerCell;
};

ConeSplit::ConeSplit(MeshPtr mesh, VertexPtr vert) :
    MeshOperation{mesh}, centerVertex{vert} {
}

HRESULT ConeSplit::apply() {
}

float ConeSplit::energy() const {
}

bool ConeSplit::depends(const TrianglePtr) const {
}

bool ConeSplit::depends(const VertexPtr) const {
}

bool ConeSplit::equals(const Edge& e) const {
}

void ConeSplit::mark() const {
}
