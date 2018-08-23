/*
 * SkeletalEdge.cpp
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#include <MxEdge.h>
#include "MeshRelationships.h"



static MxEdgeType type;

MxEdgeType *MxEdge_Type = &type;




MxEdge::MxEdge(uint id) : MxObject(MxEdge_Type), id{id}
{
}

MxEdge::~MxEdge()
{
    // TODO Auto-generated destructor stub
}

void foo(MxObject *o) {
    MxEdge *x = dyn_cast<MxEdge>(o);

    if(x) {
        std::cout << "foo";
    }
}

HRESULT connectEdgeVertices(EdgePtr edge, VertexPtr v0,
        VertexPtr v1)
{
    if(edge->vertices[0] || edge->vertices[1]) {
        return mx_error(E_FAIL, "edge vertices no null");
    }

    int v0_edges = v0->edgeCount();
    int v1_edges = v1->edgeCount();

    if(v0_edges >= 4) {
        return mx_error(E_FAIL, "vertex 1 already has 4 edges");
    }

    if(v1_edges >= 4) {
        return mx_error(E_FAIL, "vertex 2 already has 4 edges");
    }

    edge->vertices[0] = v0;
    edge->vertices[1] = v1;
    return S_OK;
}




bool MxEdge::matches(CVertexPtr a, CVertexPtr b) const
{
    return ((MxVertex*)vertices[0] == a && (MxVertex*)vertices[1] == b) ||
            ((MxVertex*)vertices[1] == a && (MxVertex*)vertices[0] == b);
}

HRESULT reconnectEdgeVertex(EdgePtr edge, VertexPtr newVertex,
        CVertexPtr oldVertex)
{
    if(edge->vertices[0] == oldVertex) {
        edge->vertices[0] = newVertex;
        return S_OK;
    }

    if(edge->vertices[1] == oldVertex) {
        edge->vertices[1] = newVertex;
        return S_OK;
    }

    return mx_error(E_INVALIDARG, "edge is not attached to the old vertex");
}

HRESULT MxEdge::erasePolygon(CPolygonPtr poly)
{
    int start = -1;
    for(int i = 0; i < SKELETAL_EDGE_MAX_TRIANGLES; ++i) {
        if(polygons[i] == poly) {
            start = i;
            break;
        }
    }

    if(start == -1) {
        return mx_error(E_INVALIDARG, "polygon is not attached to this edge");
    }

    for(int i = start; i < SKELETAL_EDGE_MAX_TRIANGLES; ++i) {

        if(i < SKELETAL_EDGE_MAX_TRIANGLES - 1) {
            polygons[i] = polygons[i+1];
        }
        else {
            polygons[i] = nullptr;
        }
    }
    return S_OK;
}

std::ostream& operator <<(std::ostream& os, CEdgePtr edge)
{
    os << "edge {id=" << edge->id << ", verts={";
    os << (edge->vertices[0] ? std::to_string(edge->vertices[0]->id) : "null");
    os << ", ";
    os << (edge->vertices[1] ? std::to_string(edge->vertices[1]->id) : "null");
    os << "}";
    return os;
}
