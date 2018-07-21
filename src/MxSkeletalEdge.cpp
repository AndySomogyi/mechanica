/*
 * SkeletalEdge.cpp
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#include <MxSkeletalEdge.h>

struct MxSkeletalEdgeType : MxType {

};

static MxSkeletalEdgeType type;

MxType *MxSkeletalEdge_Type = &type;




MxSkeletalEdge::MxSkeletalEdge() : MxObject(MxSkeletalEdge_Type)
{
}

MxSkeletalEdge::~MxSkeletalEdge()
{
    // TODO Auto-generated destructor stub
}

void foo(MxObject *o) {
    MxSkeletalEdge *x = dyn_cast<MxSkeletalEdge>(o);

    if(x) {
        std::cout << "foo";
    }
}

HRESULT connectEdgeVertices(SkeletalEdgePtr, SkeletalVertexPtr,
        SkeletalVertexPtr)
{
}

HRESULT disconnectEdgeVertices(SkeletalEdgePtr)
{
}

HRESULT connectEdgeTriangle(SkeletalEdgePtr, TrianglePtr)
{
}

HRESULT disconnectEdgeTriangle(SkeletalEdgePtr, TrianglePtr)
{
}

bool MxSkeletalEdge::matches(CVertexPtr a, CVertexPtr b) const
{
    return ((MxVertex*)vertices[0] == a && (MxVertex*)vertices[1] == b) ||
            ((MxVertex*)vertices[1] == a && (MxVertex*)vertices[0] == b);
}
