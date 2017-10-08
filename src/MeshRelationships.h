/*
 * MeshRelationships.h
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#ifndef SRC_MESHRELATIONSHIPS_H_
#define SRC_MESHRELATIONSHIPS_H_

#include "MxCell.h"

bool adjacent(const TrianglePtr a, const TrianglePtr b);

bool incident(const TrianglePtr t, const CellPtr c);

inline bool incident(const CellPtr c, const TrianglePtr t ) {
	return incident(t, c);
}

bool incident(const TrianglePtr tri, const struct MxVertex *v);

inline bool incident(const struct MxVertex *v, const TrianglePtr tri) {
	return incident(tri, v);
}

bool incident(const FacetPtr facet, const CellPtr cell);

inline bool incident(const CellPtr cell, const FacetPtr facet) {
	return incident(facet, cell);
}

bool incident(const VertexPtr vertex, const FacetPtr facet);

inline bool incident(const FacetPtr facet, const VertexPtr vertex) {
    return incident(vertex, facet);
}

bool adjacent(const PTrianglePtr a, PTrianglePtr b);

#endif /* SRC_MESHRELATIONSHIPS_H_ */
