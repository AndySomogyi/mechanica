/*
 * MeshRelationships.cpp
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#include <MeshRelationships.h>

bool incident(const TrianglePtr t, const CellPtr c) {
    return t->cells[0] == c || t->cells[1] == c;
}

bool incident(const TrianglePtr tri, const struct MxVertex *v)  {
    return tri->vertices[0] == v || tri->vertices[1] == v || tri->vertices[2] == v;
}

bool adjacent(const TrianglePtr a, const TrianglePtr b) {
}
