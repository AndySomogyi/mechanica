
/* Tuple object interface */

#ifndef Mx_TUPLEOBJECT_H
#define Mx_TUPLEOBJECT_H

#include <carbon.h>


#ifdef __cplusplus
extern "C" {
#endif

/*
Another generally useful object type is a tuple of object pointers.
For Mechanica, this is an immutable type.  C code can change the tuple items
(but not their number), and even use tuples are general-purpose arrays of
object references, but in general only brand new tuples should be mutated,
not ones that might already have been exposed to Mechanica code.

*** WARNING *** MxTuple_SetItem does not increment the new item's reference
count, but does decrement the reference count of the item it replaces,
if not nil.  It does *decrement* the reference count if it is *not*
inserted in the tuple.  Similarly, MxTuple_GetItem does not increment the
returned item's reference count.
*/



CAPI_DATA(struct CType) MxTuple_Type;
CAPI_DATA(struct CType) MxTupleIter_Type;

CAPI_DATA(int) MxTuple_Check(CObject*);
CAPI_DATA(int) MxTuple_CheckExact(CObject*);

CAPI_FUNC(CObject *) MxTuple_New(Mx_ssize_t size);
CAPI_FUNC(Mx_ssize_t) MxTuple_Size(CObject *);
CAPI_FUNC(CObject *) MxTuple_GetItem(CObject *, Mx_ssize_t);
CAPI_FUNC(int) MxTuple_SetItem(CObject *, Mx_ssize_t, CObject *);
CAPI_FUNC(CObject *) MxTuple_GetSlice(CObject *, Mx_ssize_t, Mx_ssize_t);
CAPI_FUNC(CObject *) MxTuple_Pack(Mx_ssize_t, ...);
CAPI_FUNC(int) MxTuple_ClearFreeList(void);


#ifdef __cplusplus
}
#endif
#endif /* !Mx_TUPLEOBJECT_H */
