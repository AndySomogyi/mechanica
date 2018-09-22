/*
 * SkeletalEdge.h
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#ifndef SRC_MXEDGE_H_
#define SRC_MXEDGE_H_

#include "MxMeshCore.h"


#define EDGE_MAX_POLYGONS 3

MxAPI_DATA(struct MxType*) MxEdge_Type;

struct MxVertex;

/**
 * A skeletal edge is at the boundary of three or more cells.
 */
struct MxEdge : MxObject
{
    const uint id;

    MxEdge(uint id);
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
    MxPolygon *polygons[EDGE_MAX_POLYGONS] = {nullptr};

    static bool classof(const MxObject *o) {
        return o->ob_type == MxEdge_Type;
    }

    static MxType *type() {return MxEdge_Type;};

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

    /**
     * searches through the list of polygons and clears the given polygon from the list,
     * and moves the remaining polygons down
     */
    HRESULT erasePolygon(CPolygonPtr poly);

    /**
     * inserts a polygon into this edge's list of polygons. Only inserts the polygon and modifies
     * the edge, does NOT alter the polygon. Caller is responsible for ensuring the polygon also
     * points to this edge.
     */
    HRESULT insertPolygon(CPolygonPtr poly);

    /**
     * replace oldPoly with newPoly in this edge's polygon list.
     *
     * return failure if oldPoly is not in this edge's polygon list.
     */
    HRESULT replacePolygon(CPolygonPtr newPoly, CPolygonPtr oldPoly);

    int polygonIndex(CPolygonPtr poly) const {
        for(int i = 0; i < EDGE_MAX_POLYGONS; ++i) {
            if(polygons[i] == poly) {
                return i;
            }
        }
        return -1;
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
 * Reconnects an edge to a different vertex.
 */
HRESULT reconnectEdgeVertex(EdgePtr edge, VertexPtr newVertex, CVertexPtr oldVertex);

std::ostream& operator<<(std::ostream& os, CEdgePtr edge);


#endif /* SRC_MXEDGE_H_ */
