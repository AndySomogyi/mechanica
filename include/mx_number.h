/*
 * mx_number.h
 *
 *  Created on: Jul 8, 2015
 *      Author: andy
 */

#ifndef _INCLUDED_MX_NUMBER_H_
#define _INCLUDED_MX_NUMBER_H_

#include "carbon.h"


#ifdef __cplusplus
extern "C"
{
#endif

CAPI_FUNC(int) MxNumber_Check(CObject *o);

/*
 Returns 1 if the object, o, provides numeric protocols, and
 false otherwise.

 This function always succeeds.
 */

CAPI_FUNC(CObject *) MxNumber_Add(CObject *o1, CObject *o2);

/*
 Returns the result of adding o1 and o2, or null on failure.
 This is the equivalent of the Mechanica expression: o1+o2.
 */

CAPI_FUNC(CObject *) MxNumber_Subtract(CObject *o1, CObject *o2);

/*
 Returns the result of subtracting o2 from o1, or null on
 failure.  This is the equivalent of the Mechanica expression:
 o1-o2.
 */

CAPI_FUNC(CObject *) MxNumber_Multiply(CObject *o1, CObject *o2);

/*
 Returns the result of multiplying o1 and o2, or null on
 failure.  This is the equivalent of the Mechanica expression:
 o1*o2.
 */

CAPI_FUNC(CObject *) MxNumber_MatrixMultiply(CObject *o1, CObject *o2);

/*
 This is the equivalent of the Mechanica expression: o1 @ o2.
 */

CAPI_FUNC(CObject *) MxNumber_FloorDivide(CObject *o1, CObject *o2);

/*
 Returns the result of dividing o1 by o2 giving an integral result,
 or null on failure.
 This is the equivalent of the Mechanica expression: o1//o2.
 */

CAPI_FUNC(CObject *) MxNumber_TrueDivide(CObject *o1, CObject *o2);

/*
 Returns the result of dividing o1 by o2 giving a float result,
 or null on failure.
 This is the equivalent of the Mechanica expression: o1/o2.
 */

CAPI_FUNC(CObject *) MxNumber_Remainder(CObject *o1, CObject *o2);

/*
 Returns the remainder of dividing o1 by o2, or null on
 failure.  This is the equivalent of the Mechanica expression:
 o1%o2.
 */

CAPI_FUNC(CObject *) MxNumber_Divmod(CObject *o1, CObject *o2);

/*
 See the built-in function divmod.  Returns NULL on failure.
 This is the equivalent of the Mechanica expression:
 divmod(o1,o2).
 */

CAPI_FUNC(CObject *) MxNumber_Power(CObject *o1, CObject *o2,
		CObject *o3);

/*
 See the built-in function pow.  Returns NULL on failure.
 This is the equivalent of the Mechanica expression:
 pow(o1,o2,o3), where o3 is optional.
 */

CAPI_FUNC(CObject *) MxNumber_Negative(CObject *o);

/*
 Returns the negation of o on success, or null on failure.
 This is the equivalent of the Mechanica expression: -o.
 */

CAPI_FUNC(CObject *) MxNumber_Positive(CObject *o);

/*
 Returns the (what?) of o on success, or NULL on failure.
 This is the equivalent of the Mechanica expression: +o.
 */

CAPI_FUNC(CObject *) MxNumber_Absolute(CObject *o);

/*
 Returns the absolute value of o, or null on failure.  This is
 the equivalent of the Mechanica expression: abs(o).
 */

CAPI_FUNC(CObject *) MxNumber_Invert(CObject *o);

/*
 Returns the bitwise negation of o on success, or NULL on
 failure.  This is the equivalent of the Mechanica expression:
 ~o.
 */

CAPI_FUNC(CObject *) MxNumber_Lshift(CObject *o1, CObject *o2);

/*
 Returns the result of left shifting o1 by o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1 << o2.
 */

CAPI_FUNC(CObject *) MxNumber_Rshift(CObject *o1, CObject *o2);

/*
 Returns the result of right shifting o1 by o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1 >> o2.
 */

CAPI_FUNC(CObject *) MxNumber_And(CObject *o1, CObject *o2);

/*
 Returns the result of bitwise and of o1 and o2 on success, or
 NULL on failure. This is the equivalent of the Mechanica
 expression: o1&o2.

 */

CAPI_FUNC(CObject *) MxNumber_Xor(CObject *o1, CObject *o2);

/*
 Returns the bitwise exclusive or of o1 by o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1^o2.
 */

CAPI_FUNC(CObject *) MxNumber_Or(CObject *o1, CObject *o2);

/*
 Returns the result of bitwise or on o1 and o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1|o2.
 */

#define MxIndex_Check(obj) \
   ((obj)->ob_type->tp_as_number != NULL && \
    (obj)->ob_type->tp_as_number->nb_index != NULL)

CAPI_FUNC(CObject *) MxNumber_Index(CObject *o);

/*
 Returns the object converted to a Mechanica int
 or NULL with an error raised on failure.
 */

CAPI_FUNC(Mx_ssize_t) MxNumber_AsSsize_t(CObject *o, CObject *exc);

/*
 Returns the object converted to Mx_ssize_t by going through
 MxNumber_Index first.  If an overflow error occurs while
 converting the int to Mx_ssize_t, then the second argument
 is the error-type to return.  If it is NULL, then the overflow error
 is cleared and the value is clipped.
 */

CAPI_FUNC(CObject *) MxNumber_Long(CObject *o);

/*
 Returns the o converted to an integer object on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: int(o).
 */

CAPI_FUNC(CObject *) MxNumber_Float(CObject *o);

/*
 Returns the o converted to a float object on success, or NULL
 on failure.  This is the equivalent of the Mechanica expression:
 float(o).
 */

/*  In-place variants of (some of) the above number protocol functions */

CAPI_FUNC(CObject *) MxNumber_InPlaceAdd(CObject *o1, CObject *o2);

/*
 Returns the result of adding o2 to o1, possibly in-place, or null
 on failure.  This is the equivalent of the Mechanica expression:
 o1 += o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceSubtract(CObject *o1, CObject *o2);

/*
 Returns the result of subtracting o2 from o1, possibly in-place or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 -= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceMultiply(CObject *o1, CObject *o2);

/*
 Returns the result of multiplying o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 *= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceMatrixMultiply(CObject *o1, CObject *o2);

/*
 This is the equivalent of the Mechanica expression: o1 @= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceFloorDivide(CObject *o1,
		CObject *o2);

/*
 Returns the result of dividing o1 by o2 giving an integral result,
 possibly in-place, or null on failure.
 This is the equivalent of the Mechanica expression:
 o1 /= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceTrueDivide(CObject *o1,
		CObject *o2);

/*
 Returns the result of dividing o1 by o2 giving a float result,
 possibly in-place, or null on failure.
 This is the equivalent of the Mechanica expression:
 o1 /= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceRemainder(CObject *o1, CObject *o2);

/*
 Returns the remainder of dividing o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 %= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlacePower(CObject *o1, CObject *o2,
		CObject *o3);

/*
 Returns the result of raising o1 to the power of o2, possibly
 in-place, or null on failure.  This is the equivalent of the Mechanica
 expression: o1 **= o2, or pow(o1, o2, o3) if o3 is present.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceLshift(CObject *o1, CObject *o2);

/*
 Returns the result of left shifting o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 <<= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceRshift(CObject *o1, CObject *o2);

/*
 Returns the result of right shifting o1 by o2, possibly in-place or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 >>= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceAnd(CObject *o1, CObject *o2);

/*
 Returns the result of bitwise and of o1 and o2, possibly in-place,
 or null on failure. This is the equivalent of the Mechanica
 expression: o1 &= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceXor(CObject *o1, CObject *o2);

/*
 Returns the bitwise exclusive or of o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 ^= o2.
 */

CAPI_FUNC(CObject *) MxNumber_InPlaceOr(CObject *o1, CObject *o2);

/*
 Returns the result of bitwise or of o1 and o2, possibly in-place,
 or null on failure.  This is the equivalent of the Mechanica
 expression: o1 |= o2.
 */

CAPI_FUNC(CObject *) MxNumber_ToBase(CObject *n, int base);

/*
 Returns the integer n converted to a string with a base, with a base
 marker of 0b, 0o or 0x prefixed if applicable.
 If n is not an int object, it is converted with MxNumber_Index first.
 */

/**
 * Parse a string and determine if it is a numeric type, if so
 * return the appropriate type.
 */
CAPI_FUNC(CObject *) MxNumber_FromString(const char* str);

#ifdef __cplusplus
}
#endif



#endif /* _INCLUDED_MX_NUMBER_H_ */
