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

bool connectedEdgeVertex(CEdgePtr edge, CVertexPtr v);




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
 * Removes an edge and vertex from a polygon, but does NOT reconnect the
 * adjacent edges.
 *
 * Returns the adjacent edge pointers in e1 and e2, where e1 is before (CCW) edge, and
 * e2 is after.
 *
 * out
 *
 */
HRESULT disconnectPolygonEdgeVertex(PolygonPtr poly, EdgePtr edge, CVertexPtr v,
        EdgePtr *e1, EdgePtr *e2);


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
HRESULT insertPolygonEdge(PolygonPtr poly, EdgePtr edge);


//HRESULT connectPolygonEdges(MeshPtr mesh, PolygonPtr poly, const std::vector<EdgePtr> &edges);


/**
 * Connects the vertices to a polygon, and searches the mesh for the edges between the
 * the vertices, and connects those to the polygon also.
 *
 * An edge must be present in the mesh between each pair of vertices.
 */
HRESULT connectPolygonVertices(MeshPtr mesh, PolygonPtr poly,
        const std::vector<VertexPtr> &vertices);


HRESULT disconnectEdgeVertexFromPolygons(EdgePtr e, CVertexPtr v);


/**
 * Collapses a polygon edge down to a vertex.
 *
 * Disconnects and edge, and it's two associated vertices from a polygon, and replaces them
 * with a single vertex. Both of the vertices in the edge get disconnected
 * from the polygon, and replaced with newVert.
 *
 * Returns the previous and next edges in prevEdge and nextEdge.
 *
 * Does not re-connect the two neighboring edges, the caller is responsible
 * for re-connecting them.
 *
 *
 * Previous configuration in polygon
 *
 * ---------- o ---------- o ------------
 *   e[i-1]  v[i]  e[i]  v[i+1]  e[i+1]
 *
 *
 *         ---------- o ----------
 *          e[i-1]  v[i]  e[i]
 *
 * in the polygon, e[i-1] remains in the same position, but the edge at e[i+1] gets
 * moved down to the e[i] position. The new vertex newVert gets stored in the
 * v[i] position. The caller is responsible for connecting

 *
 *
 */
HRESULT replacePolygonEdgeAndVerticesWithVertex(PolygonPtr poly, EdgePtr edge,
        VertexPtr newVert, EdgePtr *prevEdge, EdgePtr *nextEdge);


HRESULT getPolygonAdjacentEdges(CPolygonPtr poly, CEdgePtr edge, EdgePtr *prevEdge,
        EdgePtr *nextEdge);












#endif /* SRC_MESHRELATIONSHIPS_H_ */
