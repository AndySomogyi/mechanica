/*
 * SkeletalEdge.h
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXSKELETALEDGE_H_
#define SRC_MXSKELETALEDGE_H_

#include "MxMeshCore.h"


#define SKELETAL_EDGE_MAX_TRIANGLES 3

MxAPI_DATA(struct MxType*) MxSkeletalEdge_Type;

struct MxVertex;

/**
 * A skeletal edge is at the boundary of three or more cells.
 */
struct MxSkeletalEdge : MxObject
{

    MxSkeletalEdge();
    ~MxSkeletalEdge();

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
    MxTriangle *triangles[SKELETAL_EDGE_MAX_TRIANGLES] = {nullptr};

    static bool classof(const MxObject *o) {
        return o->ob_type == MxSkeletalEdge_Type;
    }

    uint triangleCount() const {
        return triangles[0] == nullptr ? 0 :
              (triangles[1] == nullptr ? 1 :
              (triangles[2] == nullptr ? 2 : 3));
    }
};

typedef MxSkeletalEdge* SkeletalEdgePtr;
typedef const MxSkeletalEdge *CSkeletalEdgePtr;

/**
 * Connects an edge with a pair of skeletal vertices.
 *
 * The edge must not be connected, and neither of the vertices can be
 * connected to the edge.
 *
 * Does NOT connect the triangles that are connected to the vertices,
 * the triangles must be connected with connectEdgeTriangle.
 */
HRESULT connectEdgeVertices(SkeletalEdgePtr, VertexPtr, VertexPtr);

/**
 * Disconnects an edge from a pair of vertices. This clear the vertex pointers in the
 * edge and removes the edge from the vertex edge lists.
 */
HRESULT disconnectEdgeVertices(SkeletalEdgePtr);


/**
 * Connect a skeletal edge to a triangle. Checks to make sure the skeletal edge
 * has an open triangle slot, and that the triangle has an open neighbor slot.
 * The triangle must already be connected to a pair of vertices, and those vertices
 * must match the edge's vertices. Both the triangle and the edge vertex pointers
 * must already be set. The order of connecting triangles to edges is thus
 * first connect the vertices to the edges and triangles, then connect the edges
 * to the triangles.
 */
HRESULT connectEdgeTriangle(SkeletalEdgePtr, TrianglePtr);

/**
 * Disconnects a triangle from an edge, and clears the corresponding
 * triangle and neighbor slots.
 *
 * Only clears the triangle and neighbor slots, does not re-connect the
 * triangle neighbor slots to anything else.
 */
HRESULT disconnectEdgeTriangle(SkeletalEdgePtr, TrianglePtr);

#endif /* SRC_MXSKELETALEDGE_H_ */
