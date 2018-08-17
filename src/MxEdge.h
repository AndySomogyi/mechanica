/*
 * SkeletalEdge.h
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXEDGE_H_
#define SRC_MXEDGE_H_

#include "MxMeshCore.h"


#define SKELETAL_EDGE_MAX_TRIANGLES 3

MxAPI_DATA(struct MxType*) MxSkeletalEdge_Type;

struct MxVertex;

/**
 * A skeletal edge is at the boundary of three or more cells.
 */
struct MxEdge : MxObject
{

    MxEdge();
    ~MxEdge();

    /**
     * The next and prev pointers are a skeletal vertex.
     */
    MxVertex *vertices[2] = {nullptr};

    /**
     * Is this edge between the given pair of vertices.
     */
    bool matches(CVertexPtr a, CVertexPtr b) const;

    /**
     * A skeletal either 2 or 3 incident triangles.
     *
     * We get 2 triangles when we read in a mesh, and the edges of a polygonal
     * face don't have any neighbors.
     */
    MxPolygon *polygons[SKELETAL_EDGE_MAX_TRIANGLES] = {nullptr};

    static bool classof(const MxObject *o) {
        return o->ob_type == MxSkeletalEdge_Type;
    }

    uint polygonCount() const {
        return polygons[0] == nullptr ? 0 :
              (polygons[1] == nullptr ? 1 :
              (polygons[2] == nullptr ? 2 : 3));
    }
};

typedef MxEdge* EdgePtr;
typedef const MxEdge *CEdgePtr;

/**
 * Connects an edge with a pair of skeletal vertices.
 *
 * The edge must not be connected, and neither of the vertices can be
 * connected to the edge.
 *
 * Does NOT connect the triangles that are connected to the vertices,
 * the triangles must be connected with connectEdgeTriangle.
 */
HRESULT connectEdgeVertices(EdgePtr, VertexPtr, VertexPtr);

/**
 * Disconnects an edge from a pair of vertices. This clear the vertex pointers in the
 * edge and removes the edge from the vertex edge lists.
 */
HRESULT disconnectEdgeVertices(EdgePtr);


/**
 * Connect a skeletal edge to a triangle. Checks to make sure the skeletal edge
 * has an open triangle slot, and that the triangle has an open neighbor slot.
 * The triangle must already be connected to a pair of vertices, and those vertices
 * must match the edge's vertices. Both the triangle and the edge vertex pointers
 * must already be set. The order of connecting triangles to edges is thus
 * first connect the vertices to the edges and triangles, then connect the edges
 * to the triangles.
 */
HRESULT connectEdgeTriangle(EdgePtr, PolygonPtr);

/**
 * Disconnects a triangle from an edge, and clears the corresponding
 * triangle and neighbor slots.
 *
 * Only clears the triangle and neighbor slots, does not re-connect the
 * triangle neighbor slots to anything else.
 */
HRESULT disconnectEdgeTriangle(EdgePtr, PolygonPtr);

#endif /* SRC_MXEDGE_H_ */
