/*
 * ca_list.h
 *
 *  Created on: Apr 22, 2016
 *      Author: andy
 */

#ifndef INCLUDE_CA_LIST_H_
#define INCLUDE_CA_LIST_H_

#include "MxObject.h"



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

#ifdef __cplusplus

struct MxList : MxObject {

    /* Vector of pointers to list elements.  list[0] is ob_item[0], etc. */
	MxObject **ob_item;

    /* ob_item contains space for 'allocated' elements.  The number
     * currently in use is ob_size.
     * Invariants:
     *     0 <= ob_size <= allocated
     *     len(list) == ob_size
     *     ob_item == NULL implies ob_size == allocated == 0
     * list.sort() temporarily sets allocated to -1 to detect mutations.
     *
     * Items must normally not be NULL, except during construction when
     * the list is not yet visible outside the function that builds it.
     */
    Mx_ssize_t allocated;
} ;

#endif

#ifdef __cplusplus
extern "C" {
#endif




//PyAPI_DATA(PyTypeObject) PyList_Type;
//PyAPI_DATA(PyTypeObject) PyListIter_Type;
//PyAPI_DATA(PyTypeObject) PyListRevIter_Type;
//PyAPI_DATA(PyTypeObject) PySortWrapper_Type;

#define MxList_Check(op) \
    PyType_FastSubclass(Py_TYPE(op), Py_TPFLAGS_LIST_SUBCLASS)
#define PyList_CheckExact(op) (Py_TYPE(op) == &PyList_Type)

MxAPI_FUNC(MxList *) MxList_New(Mx_ssize_t size);
MxAPI_FUNC(Mx_ssize_t) MxList_Size(MxList *);
MxAPI_FUNC(MxObject *) MxList_GetItem(MxList *, Mx_ssize_t);
MxAPI_FUNC(int) MxList_SetItem(MxList *, Mx_ssize_t, MxObject *);
MxAPI_FUNC(int) MxList_Insert(MxList *, Mx_ssize_t, MxObject *);
MxAPI_FUNC(int) MxList_Append(MxList *, MxObject *);
MxAPI_FUNC(MxObject *) MxList_GetSlice(MxList *, Mx_ssize_t, Mx_ssize_t);
MxAPI_FUNC(int) MxList_SetSlice(MxList *, Mx_ssize_t, Mx_ssize_t, MxObject *);
MxAPI_FUNC(int) MxList_Sort(MxList *);
MxAPI_FUNC(int) MxList_Reverse(MxList *);
MxAPI_FUNC(MxObject *) MxList_AsTuple(MxList *);

MxAPI_FUNC(MxObject *) _MxList_Extend(MxList *, MxObject *);

MxAPI_FUNC(int) MxList_ClearFreeList(void);
MxAPI_FUNC(void) _MxList_DebugMallocStats(FILE *out);


/* Macro, trading safety for speed */

#define MxList_GET_ITEM(op, i) (((MxListObject *)(op))->ob_item[i])
#define MxList_SET_ITEM(op, i, v) (((MxListObject *)(op))->ob_item[i] = (v))
#define MxList_GET_SIZE(op)    Mx_SIZE(op)
#define _MxList_ITEMS(op)      (((MxListObject *)(op))->ob_item)


#ifdef __cplusplus
}
#endif


#endif /* INCLUDE_CA_LIST_H_ */
