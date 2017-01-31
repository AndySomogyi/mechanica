/*
 * ca_string.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_MX_STRING_H_
#define _INCLUDE_MX_STRING_H_

#include <mx_object.h>
#include <mx_port.h>


struct MxString;


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return true if the object o is a string object or an instance of a subtype of the string type.
 */
MxAPI_FUNC(int) MxString_Check(MxObject *o);

/**
 * Return value: New reference.
 * Return a new string object with a copy of the string v as value on success, and NULL on failure.
 * The parameter v must not be NULL; it will not be checked.
 */
MxAPI_FUNC(MxObject*) MxString_FromString(const char *v);

/**
 * Return value: New reference.
 * Return a new string object with a copy of the string v as value and length len on success,
 * and NULL on failure. If v is NULL, the contents of the string are uninitialized.
 */
MxAPI_FUNC(MxObject*) MxString_FromStringAndSize(const char *v, Mx_ssize_t len);

/**
 * Return a NUL-terminated representation of the contents of string. The
 * pointer refers to the internal buffer of string, not a copy. The data
 * must not be modified in any way, unless the string was just created using
 * PyString_FromStringAndSize(NULL, size). It must not be deallocated.
 * If string is a Unicode object, this function computes the default
 * encoding of string and operates on that. If string is not a string object
 * at all, PyString_AsString() returns NULL and raises TypeError.
 */
MxAPI_FUNC(const char*) MxString_AsString(MxObject *string);


#ifdef __cplusplus
}
#endif


#endif /* _INCLUDE_MX_STRING_H_ */
