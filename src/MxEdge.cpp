/*
 * SkeletalEdge.cpp
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#include <MxEdge.h>
#include "MeshRelationships.h"



static CType edgeType{.tp_name="MxEdge", .tp_base=CObject_Type} ;
CType *MxEdge_Type = &edgeType;




MxEdge::MxEdge(uint id) : CObject{0, MxEdge_Type}, id{id}
{
}

MxEdge::~MxEdge()
{
    // TODO Auto-generated destructor stub
}

void foo(CObject *o) {
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
    for(int i = 0; i < EDGE_MAX_POLYGONS; ++i) {
        if(polygons[i] == poly) {
            start = i;
            break;
        }
    }

    if(start == -1) {
        return mx_error(E_INVALIDARG, "polygon is not attached to this edge");
    }

    for(int i = start; i < EDGE_MAX_POLYGONS; ++i) {

        if(i < EDGE_MAX_POLYGONS - 1) {
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
    os << "}, polygons: {";
    os << (edge->polygons[0] ? std::to_string(edge->polygons[0]->id) : "null");
    os << ", ";
    os << (edge->polygons[1] ? std::to_string(edge->polygons[1]->id) : "null");
    os << ", ";
    os << (edge->polygons[2] ? std::to_string(edge->polygons[2]->id) : "null");
    os << "}}";
    return os;
}

HRESULT MxEdge::replacePolygon(CPolygonPtr newPoly, CPolygonPtr oldPoly)
{
    for(int i = 0; i < 3; ++i) {
        if(polygons[i] == oldPoly) {
            polygons[i] = const_cast<PolygonPtr>(newPoly);
            return S_OK;
        }
    }
    return mx_error(E_FAIL, "old polygon is not is this edge's polygon list");
}

HRESULT MxEdge::insertPolygon(CPolygonPtr poly) {
    return connectEdgePolygonPointers(this, const_cast<PolygonPtr>(poly));
}
