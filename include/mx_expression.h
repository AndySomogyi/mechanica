/*
 * mx_expression.h
 *
 *  Created on: Apr 3, 2018
 *      Author: andy
 */

#ifndef INCLUDE_MX_EXPRESSION_H_
#define INCLUDE_MX_EXPRESSION_H_

#include "mx_port.h"
#include "mx_object.h"
#include "mx_symbol.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 */

MxAPI_STRUCT(MxExpression);

/**
 * This instance of MxTypeObject represents the Mechanica module type.
 */
MxAPI_DATA(struct MxType*) MxExpression_Type;

/**
 * Return true if p is a list object or an instance of a subtype of the list type.
 */
int MxExpression_Check(const MxObject *p);

/**
 * Returns a borrowed reference to the head of the expression.
 */
const MxObject *MxExpression_Head(const MxExpression *p);

/**
 * Return true if p is a list object, but not an instance of a subtype of the list type.
 */
int MxExpression_CheckExact(const MxObject *p);

//MxObject* MxExpression_New(Mx_ssize_t len)
//Return value: New reference.
//Return a new list of length len on success, or NULL on failure.

//Note If len is greater than zero, the returned list object’s items are set to NULL.
// Thus you cannot use abstract API functions such as PySequence_SetItem() or expose the
// object to Python code before setting all items to a real object with MxExpression_SetItem().

/**
 * Return the length of the list object in list; this is equivalent to len(list) on a list object.
 */
Mx_ssize_t MxExpression_Size(MxObject *list);

MxExpression *MxExpression_New(MxSymbol *head, int len, ...);

/**
 * Returns a borrowed reference to the first item in the list.
 */
MxObject *MxExpression_First(const MxExpression *ex);

/**
 * Returns a new expression that contains the same head, and the list
 * items after the first item.
 */
MxExpression *MxExpression_Rest(const MxExpression *ex);

//Mx_ssize_t MxExpression_GET_SIZE(MxObject *list)
//Macro form of MxExpression_Size() without error checking.

/**
 * Return value: Borrowed reference.
 * Return the object at position index in the list pointed to by list. The position must be positive,
 * indexing from the end of the list is not supported.
 * If index is out of bounds, return NULL and set an IndexError exception.
 */
MxObject* MxExpression_GetItem(MxObject *list, Mx_ssize_t index);

//MxObject* MxExpression_GET_ITEM(MxObject *list, Mx_ssize_t i)
//Return value: Borrowed reference.
//Macro form of MxExpression_GetItem() without error checking.

/**
 * Set the item at index index in list to item. Return 0 on success or -1 on failure.
 *
 * Note This function “steals” a reference to item and discards a reference to an
 * item already in the list at the affected position.
 */
int MxExpression_SetItem(MxObject *list, Mx_ssize_t index, MxObject *item);

//
// void MxExpression_SET_ITEM(MxObject *list, Mx_ssize_t i, MxObject *o);
// Macro form of MxExpression_SetItem() without error checking. This is normally only used to fill in new lists where there is no previo//us content.

// Note This macro “steals” a reference to item, and, unlike MxExpression_SetItem(),
// does not discard a reference to any item that is being replaced; any reference
// in list at position i will be leaked.

//**
// * Insert the item item into list list in front of index index. Return 0 if successful;
// * return -1 and set an exception if unsuccessful. Analogous to list.insert(index, item).
// */
//int MxExpression_Insert(MxObject *list, Mx_ssize_t index, MxObject *item);

//**
// * Append the object item at the end of list list. Return 0 if successful;
// * return -1 and set an exception if unsuccessful. Analogous to list.append(item).
// */
//int MxExpression_Append(MxObject *list, MxObject *item);

/**
 * Return value: New reference.
 * Return a list of the objects in list containing the objects between low and high.
 * Return NULL and set an exception if unsuccessful. Analogous to list[low:high].
 * Negative indices, as when slicing from Python, are not supported.
 */
MxObject* MxExpression_GetSlice(MxObject *list, Mx_ssize_t low, Mx_ssize_t high);

/**
 * Set the slice of list between low and high to the contents of itemlist.
 * Analogous to list[low:high] = itemlist. The itemlist may be NULL, indicating the
 * assignment of an empty list (slice deletion). Return 0 on success, -1 on failure.
 * Negative indices, as when slicing from Python, are not supported.
 */
int MxExpression_SetSlice(MxObject *list, Mx_ssize_t low, Mx_ssize_t high, MxObject *itemlist);

//int MxExpression_Sort(MxObject *list)
//Sort the items of list in place. Return 0 on success, -1 on failure. This is equivalent to list.sort().

//int MxExpression_Reverse(MxObject *list)
//Reverse the items of list in place. Return 0 on success, -1 on failure. This is the equivalent of list.reverse().

//MxObject* MxExpression_AsTuple(MxObject *list)
//Return value: New reference.
//Return a new tuple object containing the contents of list; equivalent to tuple(list).

//int MxExpression_ClearFreeList()
//Clear the free list. Return the total number of freed items.

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_MX_EXPRESSION_H_ */
