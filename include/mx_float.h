/*
 * ca_string.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_CA_FLOAT_H_
#define _INCLUDE_CA_FLOAT_H_

#include <mx_port.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 Floating Point Objects
 MxFloatObject
 This subtype of MxObject represents a Mxthon floating point object.

 MxTypeObject MxFloat_Type
 This instance of MxTypeObject represents the Mxthon floating point type. This
 is the same object as float and types.FloatType.
 */

/**
 * Return true if its argument is a MxFloatObject or a subtype of MxFloatObject.
 * Allows subtypes to be accepted.
 */
MxAPI_FUNC(int) MxFloat_Check(MxObject *p);

/**
 * Return true if its argument is a MxFloatObject, but not a subtype of
 * MxFloatObject.
 */
MxAPI_FUNC(int) MxFloat_CheckExact(MxObject *p);

/**
 * Return value: New reference.
 * Create a MxFloatObject object based on the string value in str, or NULL on
 * failure.
 */
MxAPI_FUNC(MxObject*) MxFloat_FromString(const char *str);

/**
 * Return value: New reference.
 Create a MxFloatObject object from v, or NULL on failure.

 */
MxAPI_FUNC(MxObject*) MxFloat_FromDouble(double v);

/**
 * Return a C double representation of the contents of Mxfloat.
 *
 * If Mxfloat is not a Mxthon floating point object but has a __float__()
 * method, this method will first be called to convert Mxfloat into a float.
 * This method returns -1.0 upon failure, so one should call
 * MxErr_Occurred() to check for errors.
 */
MxAPI_FUNC(double) MxFloat_AsDouble(MxObject *p);

/**
 * Return a structseq instance which contains information about the precision,
 * minimum and maximum values of a float. It’s a thin wrapper around the header
 * file float.h.
 */
MxAPI_FUNC(MxObject*) MxFloat_GetInfo(void);

/**
 * Return the maximum representable finite float DBL_MAX as C double.
 */
MxAPI_FUNC(double) MxFloat_GetMax();

/**
 * Return the minimum normalized positive float DBL_MIN as C double.
 */
MxAPI_FUNC(double) MxFloat_GetMin();

#ifdef __cplusplus
}
#endif

#endif /* _INCLUDE_CA_FLOAT_H_ */
