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
    // TODO Auto-generated constructor stub

}

MxSkeletalVertex::~MxSkeletalVertex()
{
    // TODO Auto-generated destructor stub
}

