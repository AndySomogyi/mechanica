/*
 * ca_type.h
 *
 *  Created on: Jul 3, 2015
 *      Author: andy
 */

#ifndef PYCALC_INCLUDE_CA_TYPE_H_
#define PYCALC_INCLUDE_CA_TYPE_H_

#include "mx_port.h"
#include "mx_object.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * Keep type description opaque for now, this is essentially a vtable with
 * additional metadata.
 */
typedef struct MxType MxType;

/**
 * Return true if the object o is a type object, including instances of types
 * derived from the standard type object. Return false in all other cases.
 */
MxAPI_FUNC(int) MxType_Check(MxObject *o);

/**
 Return true if the object o is a type object, but not a subtype
 of the standard type object. Return false in all other cases.
 */
MxAPI_FUNC(int) MxType_CheckExact(MxObject *o);

/**
 Clear the internal lookup cache. Return the current version tag.
 */
MxAPI_FUNC(unsigned int) MxType_ClearCache();

/**
 Invalidate the internal lookup cache for the type and all of its subtypes.
 This function must be called after any manual modification of the attributes or base classes of the type.
 */
MxAPI_FUNC(void) MxType_Modified(MxType *type);

/**
 * Return true if the type object o sets the feature feature.
 * Type features are denoted by single bit flags.
 */
MxAPI_FUNC(int) MxType_HasFeature(MxObject *o, int feature);

/**
 * Return true if a is a subtype of b.
 *
 * This function only checks for actual subtypes, which means that
 * __subclasscheck__() is not called on b. Mxll MxObject_IsSubclass()
 * to do the same check that issubclass() would do.
 */
MxAPI_FUNC(int) MxType_IsSubtype(MxType *a, MxType *b);

/**
 * Return value: New reference.
 */
MxAPI_FUNC(MxObject*) MxType_GenericAlloc(MxType *type, size_t nitems);

/**
 * Return value: New reference.
 */
MxAPI_FUNC(MxObject*) MxType_GenericNew(MxType *type, MxObject *args,
		MxObject *kwds);



#ifdef __cplusplus
}
#endif

#endif /* PYCALC_INCLUDE_CA_TYPE_H_ */
