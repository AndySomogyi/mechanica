/*
 * ca_function.h
 *
 *  Created on: Aug 3, 2015
 *      Author: andy
 *
 * Function Objects
 * There are a few functions specific to Python functions.
 */

#ifndef _INCLUDED_CA_FUNCTION_H_
#define _INCLUDED_CA_FUNCTION_H_

#include "mx_object.h"

#ifdef __cplusplus
extern "C"
{
#endif


/**
 * Return true if o is a function object (has type MxFunction_Type). The parameter
 * must not be NULL.
 */
MxAPI_FUNC(int) MxFunction_Check(MxObject *o);

/**
 * Return value: New reference.
 * Return a new function object associated with the code object code. globals must
 * be a dictionary with the global variables accessible to the function.
 *
 * The function’s docstring, name and __module__ are retrieved from the code object,
 * the argument defaults and closure are set to NULL.
 */
MxAPI_FUNC(MxObject*) MxFunction_New(MxObject *code, MxObject *globals);


/**
 * Return value: New reference.
 * As MxFunction_New(), but also allows to set the function object’s __qualname__
 * attribute. qualname should be a unicode object or NULL; if NULL, the
 * __qualname__ attribute is set to the same value as its __name__ attribute.
 */
MxAPI_FUNC(MxObject*) MxFunction_NewWithQualName(MxObject *code, MxObject *globals,
		MxObject *qualname);

/**
 * Return value: Borrowed reference.
 * Return the code object associated with the function object op.
 */
MxAPI_FUNC(MxObject*) MxFunction_GetCode(MxObject *op);

/**
 * Return value: Borrowed reference.
 * Return the globals dictionary associated with the function object op.
 */
MxAPI_FUNC(MxObject*) MxFunction_GetGlobals(MxObject *op);

/**
 * Return value: Borrowed reference.
 * Return the __module__ attribute of the function object op. This is normally a
 * string containing the module name, but can be set to any other object by
 * Mxthon code.
 */
MxAPI_FUNC(MxObject*) MxFunction_GetModule(MxObject *op);

/**
 * Return value: Borrowed reference.
 * Return the argument default values of the function object op. This can be a
 * tuple of arguments or NULL.
 */
MxAPI_FUNC(MxObject*) MxFunction_GetDefaults(MxObject *op);

/**
 * Set the argument default values for the function object op. defaults must be
 * Mx_None or a tuple.
 * Raises SystemError and returns -1 on failure.
 */
MxAPI_FUNC(int) MxFunction_SetDefaults(MxObject *op, MxObject *defaults);

/**
 * Return value: Borrowed reference.
 * Return the closure associated with the function object op. This can be NULL or
 * a tuple of cell objects.
 */
MxAPI_FUNC(MxObject*) MxFunction_GetClosure(MxObject *op);

/**
 * Set the closure associated with the function object op. closure must be Mx_None
 * or a tuple of cell objects.
 *
 * Raises SystemError and returns -1 on failure.
 */
MxAPI_FUNC(int) MxFunction_SetClosure(MxObject *op, MxObject *closure);

/**
 * Return the annotations of the function object op. This can be a mutable
 * dictionary or NULL.
 */
MxAPI_FUNC(MxObject*) MxFunction_GetAnnotations(MxObject *op);

/**
 * Set the annotations for the function object op. annotations must be a dictionary
 * or Mx_None.
 *
 * Raises SystemError and returns -1 on failure.
 */
MxAPI_FUNC(int) MxFunction_SetAnnotations(MxObject *op, MxObject *annotations);


#ifdef __cplusplus
}
#endif

#endif /* _INCLUDED_CA_FUNCTION_H_ */
