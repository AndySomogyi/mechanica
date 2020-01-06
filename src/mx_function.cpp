/*
 * ca_function.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: andy
 */
#include "mechanica_private.h"


/// PUBLIC API SECTION
extern "C" {

int MxFunction_Check(CObject* o)
{
	return 0;
}

CObject* MxFunction_New(CObject* code, CObject* globals)
{
	return 0;
}

CObject* MxFunction_NewWithQualName(CObject* code, CObject* globals, CObject* qualname)
{
	return 0;
}

CObject* MxFunction_GetCode(CObject* op)
{
	return 0;
}

CObject* MxFunction_GetGlobals(CObject* op)
{
	return 0;
}

CObject* MxFunction_GetModule(CObject* op)
{
	return 0;
}

CObject* MxFunction_GetDefaults(CObject* op)
{
	return 0;
}

int MxFunction_SetDefaults(CObject* op, CObject* defaults)
{
	return 0;
}

CObject* MxFunction_GetClosure(CObject* op)
{
	return 0;
}

int MxFunction_SetClosure(CObject* op, CObject* closure)
{
	return 0;
}

CObject* MxFunction_GetAnnotations(CObject* op)
{
	return 0;
}

int MxFunction_SetAnnotations(CObject* op, CObject* annotations)
{
	return 0;
}

}
