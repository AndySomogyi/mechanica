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

/**
 * T1 : MxObject
 * T2 : T1
 * T3 : T2
 * T4 : T1
 *
 *
 
 */

MxAPI_FUNC(int) MxType_IsSubtype(MxType *a, MxType *b) {
    do {
        if (a == b)
            return 1;
        a = a->tp_base;
    } while (a != NULL && a != MxObject_Type);
    return b == MxObject_Type;
}
