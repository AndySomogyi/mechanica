/*
 * MxTriangle.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include "MxTriangle.h"
#include "MxCell.h"


int MxTriangle::matchVertexIndices(const std::array<VertexPtr, 3> &indices) {
    typedef std::array<VertexPtr, 3> vertind;

    if (vertices == indices ||
        vertices == vertind{{indices[1], indices[2], indices[0]}} ||
        vertices == vertind{{indices[2], indices[0], indices[1]}}) {
        return 1;
    }

    if (vertices == vertind{{indices[2], indices[1], indices[0]}} ||
        vertices == vertind{{indices[1], indices[0], indices[2]}} ||
        vertices == vertind{{indices[0], indices[2], indices[1]}}) {
        return -1;
    }
    return 0;
}


MxTriangle::MxTriangle(MxTriangleType* type,
        const std::array<VertexPtr, 3>& verts,
        const std::array<CellPtr, 2>& cells,
        const std::array<MxPartialTriangleType*, 2>& partTriTypes,
        FacetPtr facet) :
            MxObject{type}, vertices{verts}, cells{cells},
            partialTriangles{{{partTriTypes[0], this}, {partTriTypes[1], this}}},
            facet{facet} {
    for(VertexPtr vert : verts) {
        auto res = vert->appendTriangle(this);
        assert(res==S_OK);
    }

    positionsChanged();
}

int MxTriangle::adjacentEdgeIndex(const VertexPtr a, const VertexPtr b) const {
    for(int i = 0; i < 3; ++i) {
        if((vertices[i] == a && vertices[(i+1)%3] == b) ||
           (vertices[i] == b && vertices[(i+1)%3] == a)) {
            return i;
        }
    }
    return -1;
}

HRESULT MxTriangle::positionsChanged() {

    const Vector3& v1 = vertices[0]->position;
    const Vector3& v2 = vertices[1]->position;
    const Vector3& v3 = vertices[2]->position;

    // the aspect ratio
    float a = (v1 - v2).length();
    float b = (v2 - v3).length();
    float c = (v3 - v1).length();
    float s = (a + b + c) / 2.0;
    aspectRatio = (a * b * c) / (8.0 * (s - a) * (s - b) * (s - c));

    // A surface normal for a triangle can be calculated by taking the vector cross product
    // of two edges of that triangle. The order of the vertices used in the calculation will
    // affect the direction of the normal (in or out of the face w.r.t. winding).

    // So for a triangle p1, p2, p3, if the vector U = p2 - p1 and the
    // vector V = p3 - p1 then the normal N = U x V and can be calculated by:

    // Nx = UyVz - UzVy
    // Ny = UzVx - UxVz
    // Nz = UxVy - UyVx
    // non-normalized normal vector
    // multiply by neg 1, CCW winding.
    Vector3 abnormal = Math::normal(v1, v2, v3);
    float len = abnormal.length();
    area = 0.5 * len;
    normal = abnormal / len;

    // average position of 3 position vectors
    centroid = (v1 + v2 + v3) / 3;

    // TODO: change vertex mass only in response to some sort of mass change
    // event -- we're mass conserving.
    for(int i = 0; i < 3; ++i) {
        vertices[i]->area += area / 3.;
        vertices[i]->mass += mass / 3.;
    }

    return S_OK;
}


bool MxTriangle::isConnected() {

    for(int i = 0; i < 2; ++i) {

        if(cells[i]->isRoot()) continue;

        PTrianglePtr t = &partialTriangles[i];

        // check pointers
        if (!adjacent(t, t->neighbors[0]) ||
            !adjacent(t, t->neighbors[1]) ||
            !adjacent(t, t->neighbors[2])) {
            return false;
        }

        assert(this == t->triangle);

        // check vertices
        if (!adjacent(this, t->neighbors[0]->triangle) ||
            !adjacent(this, t->neighbors[1]->triangle) ||
            !adjacent(this, t->neighbors[2]->triangle)) {
            return false;
        }
    }
    return true;

}

bool MxTriangle::isValid() {
    return facet &&
            contains(facet->triangles, this) &&
            cells[0] && incident(this, cells[0]) &&
            cells[1] && incident(this, cells[1]) &&
            isConnected() &&
            area > 0 && aspectRatio > 0 && mass > 0;
}
