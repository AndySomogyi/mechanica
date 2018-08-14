/*
 * MxMeshCore.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <MxPolygon.h>
#include "MxMeshCore.h"
#include "MxMesh.h"
#include "MxDebug.h"


struct MxVertexType : MxType {

};

static MxVertexType type;

MxType *MxVertex_Type = &type;

MxVertex::MxVertex(MxType* derivedType) : MxObject{derivedType}
{
}



MxVertex::MxVertex() : MxVertex{MxVertex_Type}
{
}

MxVertex::MxVertex(float mass, float area, const Magnum::Vector3 &pos) :
        MxObject{MxVertex_Type},
        mass{mass}, area{area}, position{pos} {
};



std::ostream& operator <<(std::ostream& os, CVertexPtr v)
{
    os << "{id:" << v->id << ", pos:" << v->position << "}";
    return os;
}



int MxVertex::edgeCount() const
{
    for(int i = 0; i < 4; ++i) {
        if(edges[i] == nullptr) {
            return i;
        }
    }
    return 4;
}


