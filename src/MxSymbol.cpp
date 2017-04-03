/*
 * Symbol.cpp
 *
 *  Created on: May 23, 2016
 *      Author: andy
 */

#include "mx_symbol.h"
#include "mechanica_private.h"

extern "C" {

int MxSymbolCheck(MxObject* o) {
	return 0;
}


MxSymbol* MxSymbol_FromCString(const char* str)
{
	return 0;
}


MxSymbol* MxSymbol_FromCStringAndSize(const char* str, Mx_ssize_t len)
{
	return 0;
}

MxSymbol* MxSymbol_FromString(MxString* str)
{
	return 0;
}


const char* MxSymbol_AsCString(MxSymbol* sym)
{
	return 0;
}


MxString* MxSymbol_GetString(MxSymbol* sym)
{
	return 0;
}

}

void MxSymbol_init(PyObject *m) {

}

