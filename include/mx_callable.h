/*
 * ca_module.h
 *
 *  Created on: Jul 6, 2015
 *      Author: andy
 *
 * module functions, definitions and documentation copied from official
 * python website for python compatibility.
 */

#ifndef _INCLUDED_CA_CALLABLE_H_
#define _INCLUDED_CA_CALLABLE_H_

#include "carbon.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef CA_STRICT
struct MxCallable;
#else
typedef CObject MxCallable;
#endif




/**
 * Determine if the object, o, is callable.  Return 1 if the
 * object is callable and 0 otherwise..
 *
 * This function always succeeds.
 */
CAPI_FUNC(int) MxCallable_Check(CObject *);

/**
 * Return the raw, callable address of the specified function.
 * This is intended to be cast to a function pointer and called.
 * This may involve code generation
 *
 * This is the key function is the Mechanica runtime.
 *
 * Here, the callable object is specialized and JITed for the given
 * argument types. I.e. the functions foo(int) and foo(double) are
 * different, as the function was specialized for a different type
 * in each case.
 *
 * For example, to get a callable function pointer to a function
 * defined in a module, one would:
 * @code
 * // assuming one already has a module
 * CoObject* module;
 *
 *
 * @endcode
 *
 * @param callable: A callable CObject. This may be either a named,
 *                  a functor (an object with the __call__) method, or
 *                  a method on a an object.
 *
 * @param argTypes: A sequence of CTypeObjects packed into a tuple.
 * @returns: the raw function pointer address of the underlying
 *           native code object.
 */
CAPI_FUNC(void*) MxCallable_GetFuctionAddress(MxCallable *callable,
		CType *retType, CObject *argTypes);

/**
 * Same as MxCallable_GetFuctionAddress, except the arguments types are
 * given as variable number of C arguments.  The C arguments are provided
 * as CTypeObject * values, terminated by a NULL.
 */
CAPI_FUNC(void*) MxCallable_GetFuctionAddressObjArgs(MxCallable *callable,
		CType *retType, ...);

/**
 * Mxll a callable Mechanica object, callable, with
 * arguments and keywords arguments.  The 'args' argument can not be
 * NULL, but the 'kw' argument can be NULL.
 */
CAPI_FUNC(CObject *) MxCallable_Call(MxCallable *callable,
		CObject *args, CObject *kw);

/**
 * Compatibility macro
 */
#define CObject_Call MxCallable_Call

/**
 * Mxll a callable Mechanica object, callable_object, with
 * arguments given by the tuple, args.  If no arguments are
 * needed, then args may be NULL.  Returns the result of the
 * call on success, or NULL on failure.  This is the equivalent
 * of the Mechanica expression: o(*args).
 */
CAPI_FUNC(CObject *) MxCallable_CallObject(MxCallable *callable,
		CObject *args);

#define CObject_CallObject MxCallable_CallObject

/**
 * Mxll a callable Mechanica object, callable_object, with a
 * variable number of C arguments. The C arguments are described
 * using a mkvalue-style format string. The format may be NULL,
 * indicating that no arguments are provided.  Returns the
 * result of the call on success, or NULL on failure.  This is
 * the equivalent of the Mechanica expression: o(*args).
 */
CAPI_FUNC(CObject *) MxCallable_CallFunction(MxCallable *callable,
		const char *format, ...);

#define CObject_CallFunction MxCallable_CallFunction


/**
 * Mxll a callable Mechanica object, callable, with a
 * variable number of C arguments.  The C arguments are provided
 * as CObject * values, terminated by a NULL.  Returns the
 * result of the call on success, or NULL on failure.  This is
 * the equivalent of the Mechanica expression: o(*args).
 */
CAPI_FUNC(CObject *) MxCallable_CallFunctionObjArgs(MxCallable *callable,
		...);

/**
 * Compatibility macro, Python defined this originally as
 * PyObject_CallFunctionObjArgs
 */
#define CObject_CallFunctionObjArgs MxCallable_CallFunctionObjArgs




#ifdef __cplusplus
}
#endif

#endif /* _INCLUDED_CA_CALLABLE_H_ */
