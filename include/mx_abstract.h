#ifndef Mx_ABSTRACTOBJECT_H
#define Mx_ABSTRACTOBJECT_H

#include <mx_object.h>

#ifdef __cplusplus
extern "C"
{
#endif



/* new buffer API */

MxAPI_FUNC(MxObject *) MxObject_Format(MxObject* obj,
		MxObject *format_spec);
/*
 Takes an arbitrary object and returns the result of
 calling obj.__format__(format_spec).
 */

/* Iterators */

MxAPI_FUNC(MxObject *) MxObject_GetIter(MxObject *);
/* Takes an object and returns an iterator for it.
 This is typically a new iterator but if the argument
 is an iterator, this returns itself. */

#define MxIter_Check(obj) \
    ((obj)->ob_type->tp_iternext != NULL && \
     (obj)->ob_type->tp_iternext != &_MxObject_NextNotImplemented)

MxAPI_FUNC(MxObject *) MxIter_Next(MxObject *);
/* Takes an iterator object and calls its tp_iternext slot,
 returning the next value.  If the iterator is exhausted,
 this returns NULL without setting an exception.
 NULL with an exception means an error occurred. */


/*  Sequence protocol:*/

MxAPI_FUNC(int) MxSequence_Check(MxObject *o);

/*
 Return 1 if the object provides sequence protocol, and zero
 otherwise.

 This function always succeeds.
 */

MxAPI_FUNC(Mx_ssize_t) MxSequence_Size(MxObject *o);

/*
 Return the size of sequence object o, or -1 on failure.
 */

/* For DLL compatibility */
#undef MxSequence_Length
MxAPI_FUNC(Mx_ssize_t) MxSequence_Length(MxObject *o);
#define MxSequence_Length MxSequence_Size

MxAPI_FUNC(MxObject *) MxSequence_Concat(MxObject *o1, MxObject *o2);

/*
 Return the concatenation of o1 and o2 on success, and NULL on
 failure.   This is the equivalent of the Mechanica
 expression: o1+o2.
 */

MxAPI_FUNC(MxObject *) MxSequence_Repeat(MxObject *o, Mx_ssize_t count);

/*
 Return the result of repeating sequence object o count times,
 or NULL on failure.  This is the equivalent of the Mechanica
 expression: o1*count.
 */

MxAPI_FUNC(MxObject *) MxSequence_GetItem(MxObject *o, Mx_ssize_t i);

/*
 Return the ith element of o, or NULL on failure. This is the
 equivalent of the Mechanica expression: o[i].
 */

MxAPI_FUNC(MxObject *) MxSequence_GetSlice(MxObject *o, Mx_ssize_t i1, Mx_ssize_t i2);

/*
 Return the slice of sequence object o between i1 and i2, or
 NULL on failure. This is the equivalent of the Mechanica
 expression: o[i1:i2].
 */

MxAPI_FUNC(int) MxSequence_SetItem(MxObject *o, Mx_ssize_t i, MxObject *v);

/*
 Assign object v to the ith element of o.  Returns
 -1 on failure.  This is the equivalent of the Mechanica
 statement: o[i]=v.
 */

MxAPI_FUNC(int) MxSequence_DelItem(MxObject *o, Mx_ssize_t i);

/*
 Delete the ith element of object v.  Returns
 -1 on failure.  This is the equivalent of the Mechanica
 statement: del o[i].
 */

MxAPI_FUNC(int) MxSequence_SetSlice(MxObject *o, Mx_ssize_t i1, Mx_ssize_t i2,
		MxObject *v);

/*
 Assign the sequence object, v, to the slice in sequence
 object, o, from i1 to i2.  Returns -1 on failure. This is the
 equivalent of the Mechanica statement: o[i1:i2]=v.
 */

MxAPI_FUNC(int) MxSequence_DelSlice(MxObject *o, Mx_ssize_t i1, Mx_ssize_t i2);

/*
 Delete the slice in sequence object, o, from i1 to i2.
 Returns -1 on failure. This is the equivalent of the Mechanica
 statement: del o[i1:i2].
 */

MxAPI_FUNC(MxObject *) MxSequence_Tuple(MxObject *o);

/*
 Returns the sequence, o, as a tuple on success, and NULL on failure.
 This is equivalent to the Mechanica expression: tuple(o)
 */

MxAPI_FUNC(MxObject *) MxSequence_List(MxObject *o);
/*
 Returns the sequence, o, as a list on success, and NULL on failure.
 This is equivalent to the Mechanica expression: list(o)
 */

MxAPI_FUNC(MxObject *) MxSequence_Fast(MxObject *o, const char* m);
/*
 Return the sequence, o, as a list, unless it's already a
 tuple or list.  Use MxSequence_Fast_GET_ITEM to access the
 members of this list, and MxSequence_Fast_GET_SIZE to get its length.

 Returns NULL on failure.  If the object does not support iteration,
 raises a TypeError exception with m as the message text.
 */

#define MxSequence_Fast_GET_SIZE(o) \
    (MxList_Check(o) ? MxList_GET_SIZE(o) : MxTuple_GET_SIZE(o))
/*
 Return the size of o, assuming that o was returned by
 MxSequence_Fast and is not NULL.
 */

#define MxSequence_Fast_GET_ITEM(o, i)\
     (MxList_Check(o) ? MxList_GET_ITEM(o, i) : MxTuple_GET_ITEM(o, i))
/*
 Return the ith element of o, assuming that o was returned by
 MxSequence_Fast, and that i is within bounds.
 */

#define MxSequence_ITEM(o, i)\
    ( Mx_TYPE(o)->tp_as_sequence->sq_item(o, i) )
/* Assume tp_as_sequence and sq_item exist and that i does not
 need to be corrected for a negative index
 */

#define MxSequence_Fast_ITEMS(sf) \
    (MxList_Check(sf) ? ((MxListObject *)(sf))->ob_item \
                      : ((MxTupleObject *)(sf))->ob_item)
/* Return a pointer to the underlying item array for
 an object retured by MxSequence_Fast */

MxAPI_FUNC(Mx_ssize_t) MxSequence_Count(MxObject *o, MxObject *value);

/*
 Return the number of occurrences on value on o, that is,
 return the number of keys for which o[key]==value.  On
 failure, return -1.  This is equivalent to the Mechanica
 expression: o.count(value).
 */

MxAPI_FUNC(int) MxSequence_Contains(MxObject *seq, MxObject *ob);
/*
 Return -1 if error; 1 if ob in seq; 0 if ob not in seq.
 Use __contains__ if possible, else _CaSequence_IterSearch().
 */

/* For DLL-level backwards compatibility */
#undef MxSequence_In
MxAPI_FUNC(int) MxSequence_In(MxObject *o, MxObject *value);

/* For source-level backwards compatibility */
#define MxSequence_In MxSequence_Contains

/*
 Determine if o contains value.  If an item in o is equal to
 X, return 1, otherwise return 0.  On error, return -1.  This
 is equivalent to the Mechanica expression: value in o.
 */

MxAPI_FUNC(Mx_ssize_t) MxSequence_Index(MxObject *o, MxObject *value);

/*
 Return the first index for which o[i]=value.  On error,
 return -1.    This is equivalent to the Mechanica
 expression: o.index(value).
 */

/* In-place versions of some of the above Sequence functions. */

MxAPI_FUNC(MxObject *) MxSequence_InPlaceConcat(MxObject *o1, MxObject *o2);

/*
 Append o2 to o1, in-place when possible. Return the resulting
 object, which could be o1, or NULL on failure.  This is the
 equivalent of the Mechanica expression: o1 += o2.

 */

MxAPI_FUNC(MxObject *) MxSequence_InPlaceRepeat(MxObject *o, Mx_ssize_t count);

/*
 Repeat o1 by count, in-place when possible. Return the resulting
 object, which could be o1, or NULL on failure.  This is the
 equivalent of the Mechanica expression: o1 *= count.

 */

/*  Mapping protocol:*/

MxAPI_FUNC(int) MxMapping_Check(MxObject *o);

/*
 Return 1 if the object provides mapping protocol, and zero
 otherwise.

 This function always succeeds.
 */

MxAPI_FUNC(Mx_ssize_t) MxMapping_Size(MxObject *o);

/*
 Returns the number of keys in object o on success, and -1 on
 failure.  For objects that do not provide sequence protocol,
 this is equivalent to the Mechanica expression: len(o).
 */

/* For DLL compatibility */
#undef MxMapping_Length
MxAPI_FUNC(Mx_ssize_t) MxMapping_Length(MxObject *o);
#define MxMapping_Length MxMapping_Size

/* implemented as a macro:

 int MxMapping_DelItemString(MxObject *o, const char *key);

 Remove the mapping for object, key, from the object *o.
 Returns -1 on failure.  This is equivalent to
 the Mechanica statement: del o[key].
 */
#define MxMapping_DelItemString(O,K) MxObject_DelItemString((O),(K))

/* implemented as a macro:

 int MxMapping_DelItem(MxObject *o, MxObject *key);

 Remove the mapping for object, key, from the object *o.
 Returns -1 on failure.  This is equivalent to
 the Mechanica statement: del o[key].
 */
#define MxMapping_DelItem(O,K) MxObject_DelItem((O),(K))

MxAPI_FUNC(int) MxMapping_HasKeyString(MxObject *o, const char *key);

/*
 On success, return 1 if the mapping object has the key, key,
 and 0 otherwise.  This is equivalent to the Mechanica expression:
 key in o.

 This function always succeeds.
 */

MxAPI_FUNC(int) MxMapping_HasKey(MxObject *o, MxObject *key);

/*
 Return 1 if the mapping object has the key, key,
 and 0 otherwise.  This is equivalent to the Mechanica expression:
 key in o.

 This function always succeeds.

 */

MxAPI_FUNC(MxObject *) MxMapping_Keys(MxObject *o);

/*
 On success, return a list or tuple of the keys in object o.
 On failure, return NULL.
 */

MxAPI_FUNC(MxObject *) MxMapping_Values(MxObject *o);

/*
 On success, return a list or tuple of the values in object o.
 On failure, return NULL.
 */

MxAPI_FUNC(MxObject *) MxMapping_Items(MxObject *o);

/*
 On success, return a list or tuple of the items in object o,
 where each item is a tuple containing a key-value pair.
 On failure, return NULL.

 */

MxAPI_FUNC(MxObject *) MxMapping_GetItemString(MxObject *o,
		const char *key);

/*
 Return element of o corresponding to the object, key, or NULL
 on failure. This is the equivalent of the Mechanica expression:
 o[key].
 */

MxAPI_FUNC(int) MxMapping_SetItemString(MxObject *o, const char *key,
		MxObject *value);

/*
 Map the object, key, to the value, v.  Returns
 -1 on failure.  This is the equivalent of the Mechanica
 statement: o[key]=v.
 */



#ifdef __cplusplus
}
#endif

#endif /* Mx_ABSTRACTOBJECT_H */
