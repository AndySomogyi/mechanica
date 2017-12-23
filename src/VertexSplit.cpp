/*
 * VertexSplit.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: andy
 */

#include <VertexSplit.h>

VertexSplit::VertexSplit(MeshPtr _mesh, CVertexPtr) :
    MeshOperation(_mesh)
{
}

MeshOperation *VertexSplit::create(CVertexPtr)
{
}

HRESULT VertexSplit::apply()
{
}

float VertexSplit::energy() const
{
}

bool VertexSplit::depends(const TrianglePtr) const
{
}

bool VertexSplit::depends(const VertexPtr) const
{
}

bool VertexSplit::equals(const Edge& e) const
{
}

void VertexSplit::mark() const
{
}
