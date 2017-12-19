/*
 * MeshRelationships.h
 *
 *  Created on: Sep 29, 2017
 *      Author: andy
 */

#ifndef SRC_MESHRELATIONSHIPS_H_
#define SRC_MESHRELATIONSHIPS_H_

#include "MxCell.h"


/**
 * Are these two triangles adjacent by their vertices, i.e.
 * do these triangles have a pair of vertices in common.
 */
bool adjacent_vertices(CTrianglePtr a, CTrianglePtr b);

bool incident(CTrianglePtr t, CCellPtr c);

inline bool incident(CCellPtr c, CTrianglePtr t ) {
    return incident(t, c);
}

bool incident(CTrianglePtr tri, CVertexPtr v);

inline bool incident(CVertexPtr v, CTrianglePtr tri) {
    return incident(tri, v);
}

bool incident(CPTrianglePtr tri, CVertexPtr v);

inline bool incident(CVertexPtr v, CPTrianglePtr tri) {
    return incident(tri, v);
}


bool adjacent(CPTrianglePtr a, CPTrianglePtr b);

/**
 * Is the given triangle tri incident to the edge formed
 * by vertices a and b. The partial triangle incident if the
 * it is incident to both vertices a and b.
 */
bool incident(CTrianglePtr tri, const Edge&);

bool adjacent(CVertexPtr v1, CVertexPtr v2);


/**
 * Is the given partial triangle pt incident to the edge formed
 * by vertices a and b. The partial triangle incident if the
 * triangle that the partial triangle is attached to is incident to
 * this edge.
 */
bool incident(CPTrianglePtr pt, const Edge&);

/**
 * Connects the pointers of a pair of triangles. Searches through the
 * partial triangle relationships for empty pointer slots and hooks up
 * the neighbor pointers.
 */
void connect_triangle_partial_triangles(TrianglePtr a, TrianglePtr b);

/**
 * Connects the pointers of a pair of partial triangles. Searches
 * the neighbor pointers and hooks them up. The given pair of partial
 * triangles must have at least one empty (null) neighbor pointer
 * each.
 */
void connect_partial_triangles(PTrianglePtr a, PTrianglePtr b);

/**
 * Disconnects a pair of partial triangles, finds the adjacent
 * neighbor pointer and sets them both to null.
 */
void disconnect_partial_triangles(PTrianglePtr a, PTrianglePtr b);

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
//void disconnect(TrianglePtr tri, const Edge&);

/**
 * Disconnects a partial triangle from it's adjacent triangle that
 * is connected through the edge formed by vertices a and b.
 */
//void disconnect(PTrianglePtr pt, const Edge&);

/**
 * Disconnects a partial triangle from it's adjacent triangle that
 * is connected through the edge formed by vertices a and b,
 * and connects the new partial triangle n in its' place. The
 * old partial triangle o now has at least one open partial triangle
 * neighbor slot.
 */
void reconnect(PTrianglePtr o, PTrianglePtr n, const Edge&);

/**
 * Disconnects a triangle from triangle o that is adjacent triangle o
 * through the edge formed by vertices a and b,
 * and connects the new triangle n in its' place. The
 * old triangle o now has at least one open triangle
 * neighbor slot.
 */
void reconnect_triangles(TrianglePtr o, TrianglePtr n, const Edge&);

/**
 * disconnects a vertex from a triangle, and sets the triangle's
 * vertex slot that pointed to the vertex to null.
 *
 * Disconnects all the partial triangles that referred to the
 * two triangles that were disconnected.
 */
void disconnect_triangle_vertex(TrianglePtr tri, VertexPtr v);

void connect_triangle_vertex(TrianglePtr tri, VertexPtr v);


/**
 * Connect a triangle into a ring edge on the given edge index.
 * Searches the vertex triangle lists for matching triangles,
 * finds the ring list, and inserts the triangle into the ring.
 *
 * The given triangle must already be inserted into the list of
 * triangles on each vertex.
 *
 * @returns:
 * -1 if the triangle is already in the ring.
 * positive integer of the number of triangles in the list on success.
 *
 *
 */
int radialedge_connect_triangle(TrianglePtr tri, int edgeIndex);

int radialedge_disconnect_triangle(TrianglePtr tri, int edgeIndex);


/**
 * Connects two triangles in a manifold surface. The triangles
 * must have their cells and vertices set.
 *
 * If the triangles are not adjacent (they do not share any
 * neighboring vertices), returns failure.
 *
 * Fixes up the the radial edge pointers.
 *
 * Connects the adjTriangles and partial triangles of each triangle.
 *
 * Successful results:
 *     * The triangles share an edge, and adjTriangles is null,
 *       connects these pointers.
 *     * Triangles share an edge, but adjTriangles is not null,
 *       so sets adjTriangles to the current pair.
 *
 * Fail results:
 *     * triangles do not share an edge.
 */
HRESULT connect_triangles(TrianglePtr a, TrianglePtr b);

/**
 * Disconnects the triangle the given cell. Finds all the adjacent triangles
 * on this cell side, and clears this triangles adjacent points, and all the
 * adjacent pointers of the triangles that are adjacent to this cell.
 */
HRESULT disconnect_triangle_from_cell(TrianglePtr tri, CCellPtr cell);






#endif /* SRC_MESHRELATIONSHIPS_H_ */
