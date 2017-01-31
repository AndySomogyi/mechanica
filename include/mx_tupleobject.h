
/* Tuple object interface */

#ifndef Mx_TUPLEOBJECT_H
#define Mx_TUPLEOBJECT_H

#include <mx_object.h>


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



MxAPI_DATA(MxType) MxTuple_Type;
MxAPI_DATA(MxType) MxTupleIter_Type;

MxAPI_DATA(int) MxTuple_Check(MxObject*);
MxAPI_DATA(int) MxTuple_CheckExact(MxObject*);

MxAPI_FUNC(MxObject *) MxTuple_New(Mx_ssize_t size);
MxAPI_FUNC(Mx_ssize_t) MxTuple_Size(MxObject *);
MxAPI_FUNC(MxObject *) MxTuple_GetItem(MxObject *, Mx_ssize_t);
MxAPI_FUNC(int) MxTuple_SetItem(MxObject *, Mx_ssize_t, MxObject *);
MxAPI_FUNC(MxObject *) MxTuple_GetSlice(MxObject *, Mx_ssize_t, Mx_ssize_t);
MxAPI_FUNC(MxObject *) MxTuple_Pack(Mx_ssize_t, ...);
MxAPI_FUNC(int) MxTuple_ClearFreeList(void);


#ifdef __cplusplus
}
#endif
#endif /* !Mx_TUPLEOBJECT_H */
