/*
 * ca_object.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: andy
 */

#include "mechanica_private.h"
#include <stdarg.h>
#include <iostream>



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


