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

bool incident(const TrianglePtr tri, const VertexPtr v);

inline bool incident(const VertexPtr v, const TrianglePtr tri) {
	return incident(tri, v);
}

bool incident(const PTrianglePtr tri, const VertexPtr v);

inline bool incident(const VertexPtr v, const PTrianglePtr tri) {
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

bool adjacent(const PTrianglePtr a, const PTrianglePtr b);

/**
 * Is the given triangle tri incident to the edge formed
 * by vertices a and b. The partial triangle incident if the
 * it is incident to both vertices a and b.
 */
bool incident(const TrianglePtr tri, const Edge&);


/**
 * Is the given partial triangle pt incident to the edge formed
 * by vertices a and b. The partial triangle incident if the
 * triangle that the partial triangle is attached to is incident to
 * this edge.
 */
bool incident(const PTrianglePtr pt, const Edge&);

/**
 * Connects the pointers of a pair of triangles. Searches through the
 * partial triangle relationships for empty pointer slots and hooks up
 * the neighbor pointers.
 */
void connect(TrianglePtr a, TrianglePtr b);

/**
 * Connects the pointers of a pair of partial triangles. Searches
 * the neighbor pointers and hooks them up. The given pair of partial
 * triangles must have at least one empty (null) neighbor pointer
 * each.
 */
void connect(PTrianglePtr a, PTrianglePtr b);

/**
 * Disconnects a pair of partial triangles, finds the adjacent
 * neighbor pointer and sets them both to null.
 */
void disconnect(PTrianglePtr a, PTrianglePtr b);

/**
 * Disconnect a triangle from the edge formed by the vertices a and b.
 *
 * A triangle has two partial faces, one on each side, this function
 * disconnects both of the partial faces from their adjacent faces.
 *
 * If we look down at the edge (a,b):
 *
 *        \ | /
 *       -- * --
 *          | \   <- tri
 *
 * we can see that we may have many triangles incident to this edge,
 * but we only want to remove tri. This operation does not alter any
 * of the other incident triangles, it only detaches tri from it's
 * two adjacent partial triangles.
 */
void disconnect(TrianglePtr tri, const Edge&);

/**
 * Disconnects a partial triangle from it's adjacent triangle that
 * is connected through the edge formed by vertices a and b.
 */
void disconnect(PTrianglePtr pt, const Edge&);

/**
 * Disconnects a partial triangle from it's adjacent triangle that
 * is connected through the edge formed by vertices a and b,
 * and connects the new partial triangle n in its' place. The
 * old partial triangle o now has at least one open partial triangle
 * neighbor slot.
 */
void reconnect(PTrianglePtr o, PTrianglePtr n, const Edge&);

/**
 * disconnects a vertex from a triangle, and sets the triangle's
 * vertex slot that pointed to the vertex to null.
 *
 * Disconnects all the partial triangles that referred to the
 * two triangles that were disconnected.
 */
void disconnect(TrianglePtr tri, VertexPtr v);

void connect(TrianglePtr tri, VertexPtr v);




#endif /* SRC_MESHRELATIONSHIPS_H_ */
