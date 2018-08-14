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
bool adjacentPolygonVertices(CPolygonPtr a, CPolygonPtr b);

bool connectedPolygonPointers(CPolygonPtr a, CPolygonPtr b);

bool connectedPolygonCellPointers(CPolygonPtr t, CCellPtr c);


/**
 * If the given triangle shares vertices with the vertices in the
 * edge, returns the index of which side the edge is on. Otherwise
 * returns -1.
 */
int indexOfEdgeVertices(CSkeletalEdgePtr edge, CPolygonPtr tri);

/**
 * Does the edge vertices match a pair of vertices in a triangle.
 *
 * This only checks vertex relationships, this does NOT check that the
 * triangle and edge pointers are connected.
 */
bool incidentEdgePolygonVertices(CSkeletalEdgePtr edge, CPolygonPtr tri);

/**
 * Are the edge and triangle pointers connected. Only checks the pointers
 * relationships, but does not check vertex relationships.
 */
bool connectedEdgePolygonPointers(CSkeletalEdgePtr edge, CPolygonPtr tri);

inline bool connectedCellPolygonPointers(CCellPtr c, CPolygonPtr t ) {
    return connectedPolygonCellPointers(t, c);
}


bool incidentPolygonVertex(CPolygonPtr tri, CVertexPtr v);

inline bool incidentVertexPolygon(CVertexPtr v, CPolygonPtr tri) {
    return incidentPolygonVertex(tri, v);
}




/**
 * Connects a polygon with a cell.
 *
 * The triangle
 */
HRESULT connectPolygonCell(PolygonPtr tri, CellPtr cell);

/**
 * Connects a triangle with a cell.
 *
 * The triangle
 */
HRESULT disconnectPolygonCell(PolygonPtr tri, CellPtr cell);




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
HRESULT connectPolygonPolygon(PolygonPtr, PolygonPtr);











#endif /* SRC_MESHRELATIONSHIPS_H_ */
