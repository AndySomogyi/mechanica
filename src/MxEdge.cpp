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




MxEdge::MxEdge() : MxObject(MxEdge_Type)
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

HRESULT disconnectEdgeVertices(EdgePtr)
{
}


bool MxEdge::matches(CVertexPtr a, CVertexPtr b) const
{
    return ((MxVertex*)vertices[0] == a && (MxVertex*)vertices[1] == b) ||
            ((MxVertex*)vertices[1] == a && (MxVertex*)vertices[0] == b);
}
