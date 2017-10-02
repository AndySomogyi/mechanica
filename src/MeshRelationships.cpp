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
