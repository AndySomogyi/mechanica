/*
 *  mechanica_private.h
 *
 *  Created on: Jul 6, 2015
 *      Author: andy
 *
 * The internal, private header file which actually specifies all of the
 * opaque cayman data structures.
 *
 * This file must never be included before the public cayman.h file,
 * as this ensures that the public api files will never have any dependancy
 * on the internal details.
 */

#ifndef CA_STRICT
#define CA_STRICT
#endif

#ifndef _INCLUDED_MECHANICA_H_
#include "Mechanica.h"
#endif

#ifndef _INCLUDED_CAYMAN_PRIVATE_H_
#define _INCLUDED_CAYMAN_PRIVATE_H_

#include <Python.h>

// Games with importing numpy and setting up function pointers.
// Only the main Mechanica python init module, mechanica.cpp defines
// MX_IMPORTING_NUMPY_ARRAY and calls import_array()
#ifndef MX_IMPORTING_NUMPY_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL MECHANICA_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#include <assert.h>

#include "MxObject.h"
#include "MxType.h"
#include "MxSymbol.h"





/**
 * Initialize the runtime eval modules (builtins, globals)
 */
int initEval();


/**
 * Shutdown the eval modules
 */
int finalizeEval();


/**
 * Initialize the AST module
 */
int _CaAstInit();


#define MX_NOTIMPLEMENTED \
	assert("Not Implemented" && 0);\
    return 0;


#endif /* _INCLUDED_CAYMAN_PRIVATE_H_ */
