/*
 * ca_float.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: andy
 */

#include <cstdio>
#include <ctype.h>

#ifdef CType
#error CType is macro
#endif

#include <carbon.h>


#include "mechanica_private.h"


// public API
extern "C" {

int CaFloat_Check(CObject* p)
{
	MX_NOTIMPLEMENTED;
}

int CaFloat_CheckExact(CObject* p)
{
    MX_NOTIMPLEMENTED;
}

CObject* CaFloat_FromString(const char* str)
{
    MX_NOTIMPLEMENTED;
}

CObject* CaFloat_FromDouble(double v)
{
    MX_NOTIMPLEMENTED;
}

double CaFloat_AsDouble(CObject* p)
{
    MX_NOTIMPLEMENTED;
}

CObject* CaFloat_GetInfo(void)
{
    MX_NOTIMPLEMENTED;
}

double CaFloat_GetMax()
{
    MX_NOTIMPLEMENTED;
}

double CaFloat_GetMin()
{
    MX_NOTIMPLEMENTED;
}

}



