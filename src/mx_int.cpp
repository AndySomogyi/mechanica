/*
 * ca_int.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: andy
 */

#include "mechanica_private.h"

extern "C" {

CObject* MxInt_FromLong(long ival)
{
	return NULL;
}

long MxInt_AsLong(CObject* io)
{
    MX_NOTIMPLEMENTED;
}

unsigned long long MxInt_AsUnsignedLongLongMask(CObject* io)
{
    MX_NOTIMPLEMENTED;
}

size_t MxInt_AsSize_t(CObject* io)
{
    MX_NOTIMPLEMENTED;
}

long MxInt_GetMax()
{
    MX_NOTIMPLEMENTED;
}

}


