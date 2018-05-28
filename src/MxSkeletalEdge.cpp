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

