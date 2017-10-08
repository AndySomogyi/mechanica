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
