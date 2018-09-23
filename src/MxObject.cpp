/*
 * ca_object.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: andy
 */

#include "mechanica_private.h"
#include <stdarg.h>
#include <iostream>

static MxType objectType{"MxObject", nullptr};
MxType* MxObject_Type = &objectType;

struct MxObjectInitializer {

    MxObjectInitializer() {
        objectType.tp_base = MxObject_Type;
    }

};

MxObjectInitializer obj;

//MxObject_Type->tp_base = MxObject_Type;






extern "C" {

void Ca_Dealloc(MxObject*)
{
}

MxObject* MxObject_GetAttrString(MxObject *o,
		const char *attr_name)
{
	return NULL;
}



MxObject * MxObject_CallMethod(MxObject* o, const char* method, const char* format, ...)
{
	return NULL;
}

MxObject * MxObject_CallMethodObjArgs(MxObject* o, MxObject* method, ...)
{
	return NULL;
}


MxAPI_FUNC(uint32_t) Ca_IncRef(MxObject* o)
{
	return 0;
}

MxAPI_FUNC(uint32_t) Ca_DecRef(MxObject* o)
{
	return 0;
}


MxObject* MxObject_Repr(MxObject* o)
{
	return 0;
}

MxObject* MxObject_Str(MxObject* o)
{
	return 0;
}


long MxObject_HashNotImplemented(MxObject *self)
{
	/*
    PyErr_Format(PyExc_TypeError, "unhashable type: '%.200s'",
                 self->ob_type->tp_name);
                 */
    return -1;
}

}

void MxObject_init(PyObject *m) {
}
;

HRESULT MxObject_ChangeType(MxObject* obj, const MxType* type)
{
    obj->ob_type = const_cast<MxType*>(type);
    return S_OK;
}
