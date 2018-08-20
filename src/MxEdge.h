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

struct MxEdgeType : MxType {
};

MxAPI_DATA(struct MxEdgeType*) MxEdge_Type;

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
     * A skeletal either 2 or 3 incident polygons.
     *
     * We get 2 polygons when we read in a mesh, and the edges of a polygonal
     * face don't have any neighbors.
     */
    MxPolygon *polygons[SKELETAL_EDGE_MAX_TRIANGLES] = {nullptr};

    static bool classof(const MxObject *o) {
        return o->ob_type == MxEdge_Type;
    }

    uint polygonCount() const {
        return polygons[0] == nullptr ? 0 :
              (polygons[1] == nullptr ? 1 :
              (polygons[2] == nullptr ? 2 : 3));
    }

    uint vertexCount() const {
        return vertices[0] == nullptr ? 0 :
              (vertices[1] == nullptr ? 1 : 2);
    }

    friend HRESULT connectPolygonVertices(MeshPtr mesh, PolygonPtr poly,
            const std::vector<VertexPtr> &vertices);

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


#endif /* SRC_MXEDGE_H_ */
