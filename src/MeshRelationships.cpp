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
}

bool incident(const FacetPtr facet, const CellPtr cell) {
	return facet->cells[0] == cell || facet->cells[1] == cell;
}

bool adjacent(const PTrianglePtr a, PTrianglePtr b) {
    if (!a || !b || a == b) {
        return false;
    }

    TrianglePtr ta = a->triangle;
    TrianglePtr tb = a->triangle;

    for(int k = 0; k < 3; ++k) {
        if ((ta->vertices[0] == tb->vertices[k] &&
                (ta->vertices[1] == tb->vertices[(k+1)%3] ||
                 ta->vertices[1] == tb->vertices[(k+2)%3] ||
                 ta->vertices[2] == tb->vertices[(k+1)%3] ||
                 ta->vertices[2] == tb->vertices[(k+2)%3])) ||
            (ta->vertices[1] == tb->vertices[k] &&
                (ta->vertices[0] == tb->vertices[(k+1)%3] ||
                 ta->vertices[0] == tb->vertices[(k+2)%3] ||
                 ta->vertices[2] == tb->vertices[(k+1)%3] ||
                 ta->vertices[2] == tb->vertices[(k+2)%3])) ||
            (ta->vertices[2] == tb->vertices[k] &&
                (ta->vertices[0] == tb->vertices[(k+1)%3] ||
                 ta->vertices[0] == tb->vertices[(k+2)%3] ||
                 ta->vertices[1] == tb->vertices[(k+1)%3] ||
                 ta->vertices[1] == tb->vertices[(k+2)%3]))) {
            // make sure the adjacency is set up correctly
            assert((a->neighbors[0] == b ||
                    a->neighbors[1] == b ||
                    a->neighbors[2] == b) &&
                   (b->neighbors[0] == a ||
                    b->neighbors[1] == a ||
                    b->neighbors[2] == a));
            return true;
        }
    }
    return false;
}
