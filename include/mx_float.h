/*
 * ca_string.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_CA_FLOAT_H_
#define _INCLUDE_CA_FLOAT_H_

#include <carbon.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 Floating Point Objects
 MxFloatObject
 This subtype of CObject represents a Mxthon floating point object.

 CTypeObject MxFloat_Type
 This instance of CTypeObject represents the Mxthon floating point type. This
 is the same object as float and types.FloatType.
 */

/**
 * Return true if its argument is a MxFloatObject or a subtype of MxFloatObject.
 * Allows subtypes to be accepted.
 */
CAPI_FUNC(int) MxFloat_Check(CObject *p);

/**
 * Return true if its argument is a MxFloatObject, but not a subtype of
 * MxFloatObject.
 */
CAPI_FUNC(int) MxFloat_CheckExact(CObject *p);

/**
 * Return value: New reference.
 * Create a MxFloatObject object based on the string value in str, or NULL on
 * failure.
 */
CAPI_FUNC(CObject*) MxFloat_FromString(const char *str);

/**
 * Return value: New reference.
 Create a MxFloatObject object from v, or NULL on failure.

 */
CAPI_FUNC(CObject*) MxFloat_FromDouble(double v);

/**
 * Return a C double representation of the contents of Mxfloat.
 *
 * If Mxfloat is not a Mxthon floating point object but has a __float__()
 * method, this method will first be called to convert Mxfloat into a float.
 * This method returns -1.0 upon failure, so one should call
 * MxErr_Occurred() to check for errors.
 */
CAPI_FUNC(double) MxFloat_AsDouble(CObject *p);

/**
 * Return a structseq instance which contains information about the precision,
 * minimum and maximum values of a float. It’s a thin wrapper around the header
 * file float.h.
 */
CAPI_FUNC(CObject*) MxFloat_GetInfo(void);

/**
 * Return the maximum representable finite float DBL_MAX as C double.
 */
CAPI_FUNC(double) MxFloat_GetMax();

/**
 * Return the minimum normalized positive float DBL_MIN as C double.
 */
CAPI_FUNC(double) MxFloat_GetMin();

#ifdef __cplusplus
}
#endif

#endif /* _INCLUDE_CA_FLOAT_H_ */
