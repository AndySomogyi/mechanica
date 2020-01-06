/*
 * mx_error.cpp
 *
 *  Created on: Oct 3, 2017
 *      Author: andy
 */

#include <cstdio>
#include <ctype.h>

#ifdef CType
#error CType is macro
#endif


#include "mechanica_private.h"


#ifdef CType
#error CType is macro
#endif



#include <mx_error.h>
#include <iostream>

HRESULT mx_set_error(HRESULT code, const char* msg, int line,
		const char* file, const char* func) {
	std::cerr << "error: " << code << ", " << msg << ", " << line << ", " << func << std::endl;
	return code;
}
