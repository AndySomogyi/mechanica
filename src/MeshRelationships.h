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
 * Does the edge vertices match a pair of vertices in a triangle.
 *
 * This only checks vertex relationships, this does NOT check that the
 * triangle and edge pointers are connected.
 */
bool incidentEdgePolygonVertices(CEdgePtr edge, CPolygonPtr tri);

/**
 * Are the edge and triangle pointers connected. Only checks the pointers
 * relationships, but does not check vertex relationships.
 */
bool connectedEdgePolygonPointers(CEdgePtr edge, CPolygonPtr tri);

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
 * Disconnects an edge and vertex from from a polygon.
 *
 * Removes an edge and vertex from a polygon, and re-connects
 * the remaining vertex to the next incident edge. The vertex must be attached
 * to the edge and polygon, and the edge must be attached to the polygon.
 *
 * Leaves the polygon in a valid state, but the vertex and edge are left orphaned.
 *
 */
HRESULT disconnectPolygonEdgeVertex(PolygonPtr poly, EdgePtr edge, VertexPtr v);


/**
 * Inserts an edge and vertex into a polygon after the given vertex ref.
 *
 * If ref is null, the vertex and edge get inserted at the end of the polygon.
 *
 * If poly already has one or more vertices, then vert and edge are added as a pair. In
 * this case, edge must either have one end open, and the other end connected to vert, or
 * both ends open. If one end is open, that end gets connected to the last vertex in the
 * polygon. If both ends of the edge are open, then the vertex is added to the polygon,
 * and the new edge connects the last vertex in the polygon, and the given vertex.
 *
 * This procedure will always leave the edges and vertices in a polygon in a valid state.
 * A polygon with only one vertex and edge will have the edge point back to the same vertex,
 * i.e. the edge will point to the same vertex on both sides. A polygon with two or more
 * vertices will have an edge that connects the current indexed vertex to the next one,
 * so edge at index n connects vertices at index n and n + 1. A polygon with two vertices
 * has two edges, edge 0 connects vertices 0 and 1, and edge 1 connects vertices 1 and 0.
 */
HRESULT insertPolygonEdgeVertex(PolygonPtr poly, EdgePtr edge, VertexPtr vert, CVertexPtr ref);


//HRESULT connectPolygonEdges(MeshPtr mesh, PolygonPtr poly, const std::vector<EdgePtr> &edges);


/**
 * Connects the vertices to a polygon, and searches the mesh for the edges between the
 * the vertices, and connects those to the polygon also.
 *
 * An edge must be present in the mesh between each pair of vertices.
 */
HRESULT connectPolygonVertices(MeshPtr mesh, PolygonPtr poly,
        const std::vector<VertexPtr> &vertices);












#endif /* SRC_MESHRELATIONSHIPS_H_ */
