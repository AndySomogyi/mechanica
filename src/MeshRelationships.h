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
 * Determines if a pair of triangles share a pair of vertices that
 * form a common edge. Note, this does NOT check the pointer relationships
 * only that the triangles share two vertices.
 */
bool adjacentTriangleVertices(CTrianglePtr a, CTrianglePtr b);

bool connectedTrianglePointers(CTrianglePtr a, CTrianglePtr b);

bool connectedTriangleCellPointers(CTrianglePtr t, CCellPtr c);


/**
 * If the given triangle shares vertices with the vertices in the
 * edge, returns the index of which side the edge is on. Otherwise
 * returns -1.
 */
int indexOfEdgeVertices(CSkeletalEdgePtr edge, CTrianglePtr tri);

/**
 * Does the edge vertices match a pair of vertices in a triangle.
 *
 * This only checks vertex relationships, this does NOT check that the
 * triangle and edge pointers are connected.
 */
bool incidentEdgeTriangleVertices(CSkeletalEdgePtr edge, CTrianglePtr tri);

/**
 * Are the edge and triangle pointers connected. Only checks the pointers
 * relationships, but does not check vertex relationships.
 */
bool connectedEdgeTrianglePointers(CSkeletalEdgePtr edge, CTrianglePtr tri);

inline bool connectedCellTrianglePointers(CCellPtr c, CTrianglePtr t ) {
    return connectedTriangleCellPointers(t, c);
}


bool incidentTriangleVertex(CTrianglePtr tri, CVertexPtr v);

inline bool incidentVertexTriangle(CVertexPtr v, CTrianglePtr tri) {
    return incidentTriangleVertex(tri, v);
}

bool incidentPartialTriangleVertex(CPTrianglePtr tri, CVertexPtr v);

inline bool incident(CVertexPtr v, CPTrianglePtr tri) {
    return incidentPartialTriangleVertex(tri, v);
}



/**
 * Are the pointers of a pair of partial triangles connected.
 */
bool adjacentPartialTrianglePointers(CPTrianglePtr a, CPTrianglePtr b);

/**
 * Is the given triangle tri incident to the edge formed
 * by vertices a and b. The partial triangle incident incidentPartialTriangleVertex* it is incident to both
 */
bool incident(CTrianglePtr tri, const Edge&);

bool adjacent(CVertexPtr v1, CVertexPtr v2);


/**
 * Is the given partial triangle pt incident to the ` formed
 * by vertices a and b. The partial triangle incident if the
 * triangle that the partial triangle is attached to is incident to
 *connectedCellTrianincidentPartialTriangleVertexerincidentVertexTriangle
 */
bool incident(CPTrianglePtr pt, const Edge&);

/**
 * Connects the pointers of a pair of triangles. Searches through the
 * partial triangle relationships for empty pointer slots and hooks up
 * the neighbor pointers.
 *
 * If cell is null, this will find the adjacent partial triangles based
 * on the shared edge and the shared cells. If cell is not null, this will only
 * connect the partial triangles on the side that's connected to the given cell.
 */
void connect_triangle_partial_triangles(TrianglePtr a, TrianglePtr b, CCellPtr cell = nullptr);

/**
 * Connects the pointers of a pair of partial triangles. Searches
 * the neighbor pointers and hooks them up. The given pair of partial
 * triangles must have at least one empty (null) neighbor pointer
 * each.
 */
void connectPartialTrianglePartialTriangle(PTrianglePtr a, PTrianglePtr b);

/**
 * Connects a triangle with a cell.
 *
 * The triangle
 */
HRESULT connectTriangleCell(TrianglePtr tri, CellPtr cell, int index);

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
 * disconnects a vertex from a triangle, and sets the triangle's
 * vertex slot that pointed to the vertex to null.
 *
 * Disconnects all the partial triangles that referred to the
 * two triangles that were disconnected.
 */
void disconnect_triangle_vertex(TrianglePtr tri, VertexPtr v);

/**
 * replaces the o vertex with the v vertex.
 */
HRESULT replaceTriangleVertex(TrianglePtr tri, VertexPtr o, VertexPtr v);

void connect_triangle_vertex(TrianglePtr tri, VertexPtr v);

/**
 * Connect any open neighbor slots in a triangle to the neighboring triangle
 * determined by the triangle vertices. Uses the vertices to determine
 * connect index location, so a neighboring triangle in the 0 position
 * shares vertices 0 and 1, neighboring triangle in the 1 shares vertices
 * 1 and 2, and a neighboring triangle in the 2 slot shares vertices
 * 2 and 0.
 *
 * first searches for matching vertex pointers, then hooks up the neighbor
 * pointer. The neighbor pointer must be empty in both triangles, otherwise
 * error is returned.
 */
HRESULT connectTriangleTriangle(TrianglePtr, TrianglePtr);

bool connectedEdgeVertex(CSkeletalEdgePtr, CVertexPtr);









#endif /* SRC_MESHRELATIONSHIPS_H_ */
