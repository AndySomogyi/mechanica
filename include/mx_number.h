/*
 * mx_number.h
 *
 *  Created on: Jul 8, 2015
 *      Author: andy
 */

#ifndef _INCLUDED_MX_NUMBER_H_
#define _INCLUDED_MX_NUMBER_H_

#include "mx_object.h"


#ifdef __cplusplus
extern "C"
{
#endif

MxAPI_FUNC(int) MxNumber_Check(MxObject *o);

/*
 Returns 1 if the object, o, provides numeric protocols, and
 false otherwise.

 This function always succeeds.
 */

MxAPI_FUNC(MxObject *) MxNumber_Add(MxObject *o1, MxObject *o2);

/*
 Returns the result of adding o1 and o2, or null on failure.
 This is the equivalent of the Mechanica expression: o1+o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_Subtract(MxObject *o1, MxObject *o2);

/*
 Returns the result of subtracting o2 from o1, or null on
 failure.  This is the equivalent of the Mechanica expression:
 o1-o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_Multiply(MxObject *o1, MxObject *o2);

/*
 Returns the result of multiplying o1 and o2, or null on
 failure.  This is the equivalent of the Mechanica expression:
 o1*o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_MatrixMultiply(MxObject *o1, MxObject *o2);

/*
 This is the equivalent of the Mechanica expression: o1 @ o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_FloorDivide(MxObject *o1, MxObject *o2);

/*
 Returns the result of dividing o1 by o2 giving an integral result,
 or null on failure.
 This is the equivalent of the Mechanica expression: o1//o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_TrueDivide(MxObject *o1, MxObject *o2);

/*
 Returns the result of dividing o1 by o2 giving a float result,
 or null on failure.
 This is the equivalent of the Mechanica expression: o1/o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_Remainder(MxObject *o1, MxObject *o2);

/*
 Returns the remainder of dividing o1 by o2, or null on
 failure.  This is the equivalent of the Mechanica expression:
 o1%o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_Divmod(MxObject *o1, MxObject *o2);

/*
 See the built-in function divmod.  Returns NULL on failure.
 This is the equivalent of the Mechanica expression:
 divmod(o1,o2).
 */

MxAPI_FUNC(MxObject *) MxNumber_Power(MxObject *o1, MxObject *o2,
		MxObject *o3);

/*
 See the built-in function pow.  Returns NULL on failure.
 This is the equivalent of the Mechanica expression:
 pow(o1,o2,o3), where o3 is optional.
 */

MxAPI_FUNC(MxObject *) MxNumber_Negative(MxObject *o);

/*
 Returns the negation of o on success, or null on failure.
 This is the equivalent of the Mechanica expression: -o.
 */

MxAPI_FUNC(MxObject *) MxNumber_Positive(MxObject *o);

/*
 Returns the (what?) of o on success, or NULL on failure.
 This is the equivalent of the Mechanica expression: +o.
 */

MxAPI_FUNC(MxObject *) MxNumber_Absolute(MxObject *o);

/*
 Returns the absolute value of o, or null on failure.  This is
 the equivalent of the Mechanica expression: abs(o).
 */

MxAPI_FUNC(MxObject *) MxNumber_Invert(MxObject *o);

/*
 Returns the bitwise negation of o on success, or NULL on
 failure.  This is the equivalent of the Mechanica expression:
 ~o.
 */

MxAPI_FUNC(MxObject *) MxNumber_Lshift(MxObject *o1, MxObject *o2);

/*
 Returns the result of left shifting o1 by o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1 << o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_Rshift(MxObject *o1, MxObject *o2);

/*
 Returns the result of right shifting o1 by o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1 >> o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_And(MxObject *o1, MxObject *o2);

/*
 Returns the result of bitwise and of o1 and o2 on success, or
 NULL on failure. This is the equivalent of the Mechanica
 expression: o1&o2.

 */

MxAPI_FUNC(MxObject *) MxNumber_Xor(MxObject *o1, MxObject *o2);

/*
 Returns the bitwise exclusive or of o1 by o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1^o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_Or(MxObject *o1, MxObject *o2);

/*
 Returns the result of bitwise or on o1 and o2 on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: o1|o2.
 */

#define MxIndex_Check(obj) \
   ((obj)->ob_type->tp_as_number != NULL && \
    (obj)->ob_type->tp_as_number->nb_index != NULL)

MxAPI_FUNC(MxObject *) MxNumber_Index(MxObject *o);

/*
 Returns the object converted to a Mechanica int
 or NULL with an error raised on failure.
 */

MxAPI_FUNC(Mx_ssize_t) MxNumber_AsSsize_t(MxObject *o, MxObject *exc);

/*
 Returns the object converted to Mx_ssize_t by going through
 MxNumber_Index first.  If an overflow error occurs while
 converting the int to Mx_ssize_t, then the second argument
 is the error-type to return.  If it is NULL, then the overflow error
 is cleared and the value is clipped.
 */

MxAPI_FUNC(MxObject *) MxNumber_Long(MxObject *o);

/*
 Returns the o converted to an integer object on success, or
 NULL on failure.  This is the equivalent of the Mechanica
 expression: int(o).
 */

MxAPI_FUNC(MxObject *) MxNumber_Float(MxObject *o);

/*
 Returns the o converted to a float object on success, or NULL
 on failure.  This is the equivalent of the Mechanica expression:
 float(o).
 */

/*  In-place variants of (some of) the above number protocol functions */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceAdd(MxObject *o1, MxObject *o2);

/*
 Returns the result of adding o2 to o1, possibly in-place, or null
 on failure.  This is the equivalent of the Mechanica expression:
 o1 += o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceSubtract(MxObject *o1, MxObject *o2);

/*
 Returns the result of subtracting o2 from o1, possibly in-place or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 -= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceMultiply(MxObject *o1, MxObject *o2);

/*
 Returns the result of multiplying o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 *= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceMatrixMultiply(MxObject *o1, MxObject *o2);

/*
 This is the equivalent of the Mechanica expression: o1 @= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceFloorDivide(MxObject *o1,
		MxObject *o2);

/*
 Returns the result of dividing o1 by o2 giving an integral result,
 possibly in-place, or null on failure.
 This is the equivalent of the Mechanica expression:
 o1 /= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceTrueDivide(MxObject *o1,
		MxObject *o2);

/*
 Returns the result of dividing o1 by o2 giving a float result,
 possibly in-place, or null on failure.
 This is the equivalent of the Mechanica expression:
 o1 /= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceRemainder(MxObject *o1, MxObject *o2);

/*
 Returns the remainder of dividing o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 %= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlacePower(MxObject *o1, MxObject *o2,
		MxObject *o3);

/*
 Returns the result of raising o1 to the power of o2, possibly
 in-place, or null on failure.  This is the equivalent of the Mechanica
 expression: o1 **= o2, or pow(o1, o2, o3) if o3 is present.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceLshift(MxObject *o1, MxObject *o2);

/*
 Returns the result of left shifting o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 <<= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceRshift(MxObject *o1, MxObject *o2);

/*
 Returns the result of right shifting o1 by o2, possibly in-place or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 >>= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceAnd(MxObject *o1, MxObject *o2);

/*
 Returns the result of bitwise and of o1 and o2, possibly in-place,
 or null on failure. This is the equivalent of the Mechanica
 expression: o1 &= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceXor(MxObject *o1, MxObject *o2);

/*
 Returns the bitwise exclusive or of o1 by o2, possibly in-place, or
 null on failure.  This is the equivalent of the Mechanica expression:
 o1 ^= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_InPlaceOr(MxObject *o1, MxObject *o2);

/*
 Returns the result of bitwise or of o1 and o2, possibly in-place,
 or null on failure.  This is the equivalent of the Mechanica
 expression: o1 |= o2.
 */

MxAPI_FUNC(MxObject *) MxNumber_ToBase(MxObject *n, int base);

/*
 Returns the integer n converted to a string with a base, with a base
 marker of 0b, 0o or 0x prefixed if applicable.
 If n is not an int object, it is converted with MxNumber_Index first.
 */

/**
 * Parse a string and determine if it is a numeric type, if so
 * return the appropriate type.
 */
MxAPI_FUNC(MxObject *) MxNumber_FromString(const char* str);

#ifdef __cplusplus
}
#endif



#endif /* _INCLUDED_MX_NUMBER_H_ */
