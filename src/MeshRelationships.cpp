/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>
#include <algorithm>

bool incident(const TrianglePtr t, const CellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool incident(const TrianglePtr tri, const struct MxVertex *v)  {
    return tri->vertices[0] == v || tri->vertices[1] == v || tri->vertices[2] == v;
}

bool adjacent(const TrianglePtr a, const TrianglePtr b) {
    if(a == b) {
        return false;
    }

    for(int k = 0; k < 3; ++k) {
        if ((a->vertices[0] == b->vertices[k] &&
             (a->vertices[1] == b->vertices[(k+1)%3] ||
              a->vertices[1] == b->vertices[(k+2)%3] ||
              a->vertices[2] == b->vertices[(k+1)%3] ||
              a->vertices[2] == b->vertices[(k+2)%3])) ||
            (a->vertices[1] == b->vertices[k] &&
             (a->vertices[0] == b->vertices[(k+1)%3] ||
              a->vertices[0] == b->vertices[(k+2)%3] ||
              a->vertices[2] == b->vertices[(k+1)%3] ||
              a->vertices[2] == b->vertices[(k+2)%3])) ||
            (a->vertices[2] == b->vertices[k] &&
             (a->vertices[0] == b->vertices[(k+1)%3] ||
              a->vertices[0] == b->vertices[(k+2)%3] ||
              a->vertices[1] == b->vertices[(k+1)%3] ||
              a->vertices[1] == b->vertices[(k+2)%3]))) {
                 return true;
             }
    }
    return false;
}

bool incident(const FacetPtr facet, const CellPtr cell) {
	return facet->cells[0] == cell || facet->cells[1] == cell;
}

bool adjacent(const PTrianglePtr a, PTrianglePtr b) {
    if (!a || !b || a == b) {
        return false;
    }

    return (a->neighbors[0] == b || a->neighbors[1] == b || a->neighbors[2] == b) &&
           (b->neighbors[0] == a || b->neighbors[1] == a || b->neighbors[2] == a);
}

bool incident(const VertexPtr vertex, const FacetPtr facet) {
    return contains(vertex->facets, facet);
}

void connect(TrianglePtr a, TrianglePtr b) {
    assert(adjacent(a, b));
    connect(&a->partialTriangles[0], &b->partialTriangles[0]);
    connect(&a->partialTriangles[1], &b->partialTriangles[1]);
}

void connect(PTrianglePtr a, PTrianglePtr b) {
    assert(a->triangle != b->triangle && "partial triangles are on the same triangle");

    assert((!a->neighbors[0] || !a->neighbors[1] || !a->neighbors[2])
           && "connecting partial face without empty slots");
    assert((!b->neighbors[0] || !b->neighbors[1] || !b->neighbors[2])
           && "connecting partial face without empty slots");

    for(uint i = 0; i < 3; ++i) {
        assert(a->neighbors[i] != a && a->neighbors[i] != b);
        if(!a->neighbors[i]) {
            a->neighbors[i] = b;
            break;
        }
    }

    for(uint i = 0; i < 3; ++i) {
        assert(b->neighbors[i] != a && b->neighbors[i] != b);
        if(!b->neighbors[i]) {
            b->neighbors[i] = a;
            break;
        }
    }
}

void disconnect(PTrianglePtr a, PTrianglePtr b) {
    for(uint i = 0; i < 3; ++i) {
        if(a->neighbors[i] == b) {
            a->neighbors[i] = nullptr;
            break;
        }
    }

    for(uint i = 0; i < 3; ++i) {
        if(b->neighbors[i] == a) {
            b->neighbors[i] = nullptr;
            break;
        }
    }
}

bool incident(const TrianglePtr tri, const VertexPtr v) {
    return tri->vertices[0] == v || tri->vertices[1] == v || tri->vertices[2] == v;
}

void disconnect(TrianglePtr tri, const Edge& edge) {
    assert(incident(tri, edge[0]));
    assert(incident(tri, edge[1]));
    disconnect(&tri->partialTriangles[0], edge);
    disconnect(&tri->partialTriangles[1], edge);
}

void disconnect(PTrianglePtr pt, const Edge& edge) {
    for(uint i = 0; i < 3; ++i) {
        if(pt->neighbors[i] &&
           pt->neighbors[i]->triangle &&
           incident(pt->neighbors[i]->triangle, edge[0]) &&
           incident(pt->neighbors[i]->triangle, edge[1])) {
            disconnect(pt, pt->neighbors[i]);
            return;
        }
    }
    assert(0 && "partial triangle is not adjacent to given edge");
}

bool incident(PTrianglePtr pt, const Edge& edge) {
    return incident(pt->triangle, edge);
}

bool incident(TrianglePtr tri, const std::array<VertexPtr, 2>& edge) {
    return incident(tri, edge[0]) && incident(tri, edge[1]);
}


void reconnect(PTrianglePtr o, PTrianglePtr n, const std::array<VertexPtr, 2>& edge) {
    for(uint i = 0; i < 3; ++i) {
        if(o->neighbors[i] &&
           o->neighbors[i]->triangle &&
           incident(o->neighbors[i]->triangle, edge)) {
            PTrianglePtr adj = o->neighbors[i];
            disconnect(o, adj);
            connect(n, adj);
            return;
        }
    }
    assert(0 && "partial triangle is not adjacent to given edge");

}
