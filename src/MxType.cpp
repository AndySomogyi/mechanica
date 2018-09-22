/*
 * MxType.cpp
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#include <MxType.h>


static MxType typeType{"MxType", MxObject_Type};
MxType *MxType_Type = &typeType;

void MxType_init(PyObject *m) {

}

MxType::MxType(const char* name, MxType* base) : MxObject{MxType_Type}
{
    tp_name = name;
    tp_base = base;
}



MxAPI_FUNC(int) MxType_IsSubtype(MxType *a, MxType *b) {
    do {
        if(a == b->tp_base) {
            return 1;
        }
        b = b->tp_base;
    } while(b && b != MxObject_Type);
    return 0;
}
