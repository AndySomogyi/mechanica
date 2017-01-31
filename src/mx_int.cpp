/*
 * ca_int.cpp
 *
 *  Created on: Aug 5, 2015
 *      Author: andy
 */

#include "mechanica_private.h"

extern "C" {

MxObject* MxInt_FromLong(long ival)
{
	return NULL;
}

long MxInt_AsLong(MxObject* io)
{
    MX_NOTIMPLEMENTED;
}

unsigned long long MxInt_AsUnsignedLongLongMask(MxObject* io)
{
    MX_NOTIMPLEMENTED;
}

size_t MxInt_AsSize_t(MxObject* io)
{
    MX_NOTIMPLEMENTED;
}

long MxInt_GetMax()
{
    MX_NOTIMPLEMENTED;
}

}


