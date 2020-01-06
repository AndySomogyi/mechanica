/*
 * ca_list.h
 *
 *  Created on: Apr 22, 2016
 *      Author: andy
 */

#ifndef INCLUDE_CA_LIST_H_
#define INCLUDE_CA_LIST_H_

#include <carbon.h>



/* List object interface */

/*
Another generally useful object type is an list of object pointers.
This is a mutable type: the list items can be changed, and items can be
added or removed.  Out-of-range indices or non-list objects are ignored.

*** WARNING *** MxList_SetItem does not increment the new item's reference
count, but does decrement the reference count of the item it replaces,
if not nil.  It does *decrement* the reference count if it is *not*
inserted in the list.  Similarly, MxList_GetItem does not increment the
returned item's reference count.
*/

MxAPI_STRUCT(MxList);



//PyAPI_DATA(PyTypeObject) PyList_Type;
//PyAPI_DATA(PyTypeObject) PyListIter_Type;
//PyAPI_DATA(PyTypeObject) PyListRevIter_Type;
//PyAPI_DATA(PyTypeObject) PySortWrapper_Type;

#define MxList_Check(op) \
    PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_LIST_SUBCLASS)
#define PyList_CheckExact(op) (Py_TYPE(op) == &PyList_Type)

CAPI_FUNC(MxList *) MxList_New(Mx_ssize_t size);
CAPI_FUNC(Mx_ssize_t) MxList_Size(MxList *);
CAPI_FUNC(CObject *) MxList_GetItem(MxList *, Mx_ssize_t);
CAPI_FUNC(int) MxList_SetItem(MxList *, Mx_ssize_t, CObject *);
CAPI_FUNC(int) MxList_Insert(MxList *, Mx_ssize_t, CObject *);
CAPI_FUNC(int) MxList_Append(MxList *, CObject *);
CAPI_FUNC(CObject *) MxList_GetSlice(MxList *, Mx_ssize_t, Mx_ssize_t);
CAPI_FUNC(int) MxList_SetSlice(MxList *, Mx_ssize_t, Mx_ssize_t, CObject *);
CAPI_FUNC(int) MxList_Sort(MxList *);
CAPI_FUNC(int) MxList_Reverse(MxList *);
CAPI_FUNC(CObject *) MxList_AsTuple(MxList *);

CAPI_FUNC(CObject *) _MxList_Extend(MxList *, CObject *);

CAPI_FUNC(int) MxList_ClearFreeList(void);
CAPI_FUNC(void) _MxList_DebugMallocStats(FILE *out);


/* Macro, trading safety for speed */

#define MxList_GET_ITEM(op, i) (((MxList *)(op))->ob_item[i])
#define MxList_SET_ITEM(op, i, v) (((MxList *)(op))->ob_item[i] = (v))
#define MxList_GET_SIZE(op)    Mx_SIZE(op)
#define _MxList_ITEMS(op)      (((MxList *)(op))->ob_item)



#endif /* INCLUDE_CA_LIST_H_ */
