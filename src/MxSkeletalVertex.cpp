/*
 * MxSkeletalVertex.cpp
 *
 *  Created on: Mar 20, 2018
 *      Author: andy
 */

#include <MxSkeletalVertex.h>

struct MxSkeletalVertexType : MxType {

};

static MxSkeletalVertexType type;

MxType *MxSkeletalVertex_Type = &type;



MxSkeletalVertex::MxSkeletalVertex() :  MxVertex{MxSkeletalVertex_Type}
{
}

MxSkeletalVertex::~MxSkeletalVertex()
{
}

HRESULT MxSkeletalVertex::init(float mass, float area,
        const Magnum::Vector3& pos)
{
    this->mass = mass;
    this->area = area;
    this->position = pos;
    return S_OK;
}

MxSkeletalVertex::MxSkeletalVertex(float mass, float area,
        const Magnum::Vector3& pos) :
        MxVertex{MxSkeletalVertex_Type} {

        this->mass = mass;
        this->area = area;
        this->position = pos;
};
