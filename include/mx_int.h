/*
 * ca_string.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDE_CA_INT_H_
#define _INCLUDE_CA_INT_H_

#include <carbon.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**

 Plain Integer Objects
 MxIntObject
 This subtype of CObject represents a Mxthon integer object.
 */

/**
 CTypeObject MxInt_Type
 This instance of CTypeObject represents the Mxthon plain integer type.
 This is the same object as int and types.IntType.
 */

/**
 * Return true if o is of type MxInt_Type or a subtype of MxInt_Type.
 */
CAPI_FUNC(int) MxInt_Check(CObject *o);

/**
 * Return true if o is of type MxInt_Type, but not a subtype of MxInt_Type.
 */
CAPI_FUNC(int) MxInt_CheckExact(CObject *o);

/**
 * Return value: New reference.
 * Return a new MxIntObject or MxLongObject based on the string value in str,
 * which is interpreted according to the radix in base. If pend is non-NULL,
 * *pend will point to the first character in str which follows the
 * representation of the number. If base is 0, the radix will be determined
 * based on the leading characters of str: if str starts with '0x' or '0X',
 * radix 16 will be used; if str starts with '0', radix 8 will be used;
 * otherwise radix 10 will be used. If base is not 0, it must be between
 * 2 and 36, inclusive. Leading spaces are ignored. If there are no digits,
 * ValueError will be raised. If the string represents a number too large
 * to be contained within the machine’s long int type and overflow warnings
 * are being suppressed, a MxLongObject will be returned. If overflow warnings
 * are not being suppressed, NULL will be returned in this case.
 */
CAPI_FUNC( CObject)* MxInt_FromString(const char *str, char **pend, int base);

/**
 * Return value: New reference.
 * Create a new integer object with a value of ival.
 * The current implementation keeps an array of integer objects for all integers
 * between -5 and 256, when you create an int in that range you actually just get
 * back a reference to the existing object. So it should be possible to change
 * the value of 1. I suspect the behaviour of Mxthon in this case is undefined. :-)
 */
CAPI_FUNC( CObject)* MxInt_FromLong(long ival);

/**
 * Create a new integer object with a value of ival. If the value exceeds
 * LONG_MAX, a long integer object is returned.
 */
CAPI_FUNC(CObject)* MxInt_FromSize_t(size_t ival);

/**
 * Will first attempt to cast the object to a MxIntObject, if it is not
 * already one, and then return its value. If there is an error, -1 is
 * returned, and the caller should check MxErr_Occurred() to find out
 * whether there was an error, or whether the value just happened to be -1.
 */
CAPI_FUNC( long) MxInt_AsLong(CObject *io);

/**
 * Will first attempt to cast the object to a MxIntObject or MxLongObject,
 * if it is not already one, and then return its value as unsigned long long,
 * without checking for overflow.
 */
CAPI_FUNC( unsigned long long) MxInt_AsUnsignedLongLongMask(CObject *io);

/**
 * Will first attempt to cast the object to a MxIntObject or MxLongObject,
 * if it is not already one, and then return its value as Mx_ssize_t.
 */
CAPI_FUNC(size_t) MxInt_AsSize_t(CObject *io);

/**
 * Return the system’s idea of the largest integer it can handle
 * (LONG_MAX, as defined in the system header files).
 */
CAPI_FUNC(long) MxInt_GetMax();

#ifdef __cplusplus
}
#endif

#endif /* _INCLUDE_CA_INT_H_ */
