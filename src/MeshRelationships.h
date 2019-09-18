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
 * Is an edge connected to a given vertex?
 */
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
 *   e_prev[i-1]  v[i]  e[i]  v[i+1]  e_next[i+1]
 *
 *
 * ---------- o ----------
 *   e_prev[i-1]  vn[i]  e_next[i]
 *
 * in the polygon, e[i-1] remains in the same position, but the edge at e[i+1] gets
 * moved down to the e[i] position. The new vertex newVert gets stored in the
 * v[i] position. The caller is responsible for connecting the edge and vertex pointers.
 */
HRESULT replacePolygonEdgeAndVerticesWithVertex(PolygonPtr poly, EdgePtr edge,
        VertexPtr newVert, EdgePtr *prevEdge, EdgePtr *nextEdge);


/**
 * Removes a vertex from a polygon, and replaces it with an edge and two vertices.
 *
 * Need to respect polygon winding, and insert the vertices in the correct order, so we
 * have to provide the next and prev edges where the vertices will get inserted into. This
 * is why we also pass in the two vertices in this polygon, before and after the vertex,
 * so that the new vertices will get inserted in the correct order. So, if originally, we have:
 *
 * e0:v:e1 -> e0:v0:edge:v1:e1
 *
 * Index wise, we have if e0 is before e1, i.e if index of e0 is i, we have:
 * e0[i]:v[i]:e1[i+1] -> e0[i]:v0[i]:edge[i+1]:v1[i+1]:e1[i+2]
 *
 * And if e1 is before e0: i.e if index of e1 is i, we have:
 * e1[i]:v[i]:e0[i+1] -> e1[i]:v1[i]:edge[i+1]:v0[i+1]:e0[i+2]
 *
 * Also updates the e0 and e1 edges to point to the new vertices v0 and v1, so that
 * these edges no longer reference the old vert vertex that was removed.
 *
 *
 * @param vert: the original vertex that will be removed from the polygon vertex list
 * @param edge: the new edge that will get added to the polygons edge list.
 * @param v0: the first new vertex, replaces vert in the vertex list with v0
 * @param v1: the second vertex, gets inserted into the vertex list at the i+1 position.
 */
HRESULT replacePolygonVertexWithEdgeAndVertices(PolygonPtr poly, CVertexPtr vert,
        CEdgePtr e0, CEdgePtr e1,  EdgePtr edge, VertexPtr v0, VertexPtr v1);




HRESULT getPolygonAdjacentEdges(CPolygonPtr poly, CEdgePtr edge, EdgePtr *prevEdge,
        EdgePtr *nextEdge);


/**
 * splits a polygon edge, into two, (e, en), and inserts a new vertex into the polygon.
 * The new edge, en should contain one vertex in common with the existing edge, and one
 * new vertex.
 */
HRESULT splitPolygonEdge(PolygonPtr poly, EdgePtr newEdge, EdgePtr refEdge);








#endif /* SRC_MESHRELATIONSHIPS_H_ */
