/*
 * ca_function.cpp
 *
 *  Created on: Aug 3, 2015
 *      Author: andy
 */
#include "mechanica_private.h"


/// PUBLIC API SECTION
extern "C" {

int MxFunction_Check(MxObject* o)
{
	return 0;
}

MxObject* MxFunction_New(MxObject* code, MxObject* globals)
{
	return 0;
}

MxObject* MxFunction_NewWithQualName(MxObject* code, MxObject* globals, MxObject* qualname)
{
	return 0;
}

MxObject* MxFunction_GetCode(MxObject* op)
{
	return 0;
}

MxObject* MxFunction_GetGlobals(MxObject* op)
{
	return 0;
}

MxObject* MxFunction_GetModule(MxObject* op)
{
	return 0;
}

MxObject* MxFunction_GetDefaults(MxObject* op)
{
	return 0;
}

int MxFunction_SetDefaults(MxObject* op, MxObject* defaults)
{
	return 0;
}

MxObject* MxFunction_GetClosure(MxObject* op)
{
	return 0;
}

int MxFunction_SetClosure(MxObject* op, MxObject* closure)
{
	return 0;
}

MxObject* MxFunction_GetAnnotations(MxObject* op)
{
	return 0;
}

int MxFunction_SetAnnotations(MxObject* op, MxObject* annotations)
{
	return 0;
}

}
