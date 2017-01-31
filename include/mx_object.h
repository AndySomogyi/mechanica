/*
 * ca_object.h
 *
 *  Created on: Jun 30, 2015
 *      Author: andy
 */

#ifndef _INCLUDED_CA_OBJECT_H_
#define _INCLUDED_CA_OBJECT_H_

#include <mx_port.h>

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif


/**
 * Basic opaque Mechanica object type.
 */
struct MxObject {
	struct MxType *type;
	uint32_t refcount;
};


#define Mx_REFCNT(ob)           (((MxObject*)(ob))->refcount)
#define Mx_TYPE(ob)             (((MxObject*)(ob))->type)
#define Mx_SIZE(ob)             (((MxVarObject*)(ob))->ob_size)

#define Mx_INCREF(o) { Mx_IncRef((MxObject*)(o)); }

#define Mx_DECREF(o) { Mx_DecRef((MxObject*)(o)); }

/* Safely decref `op` and set `op` to NULL, especially useful in tp_clear
 * and tp_dealloc implementations.
 *
 * Note that "the obvious" code can be deadly:
 *
 *     Mx_XDECREF(op);
 *     op = NULL;
 *
 * Typically, `op` is something like self->containee, and `self` is done
 * using its `containee` member.  In the code sequence above, suppose
 * `containee` is non-NULL with a refcount of 1.  Its refcount falls to
 * 0 on the first line, which can trigger an arbitrary amount of code,
 * possibly including finalizers (like __del__ methods or weakref callbacks)
 * coded in Mechanica, which in turn can release the GIL and allow other threads
 * to run, etc.  Such code may even invoke methods of `self` again, or cause
 * cyclic gc to trigger, but-- oops! --self->containee still points to the
 * object being torn down, and it may be in an insane state while being torn
 * down.  This has in fact been a rich historic source of miserable (rare &
 * hard-to-diagnose) segfaulting (and other) bugs.
 *
 * The safe way is:
 *
 *      Mx_CLEAR(op);
 *
 * That arranges to set `op` to NULL _before_ decref'ing, so that any code
 * triggered as a side-effect of `op` getting torn down no longer believes
 * `op` points to a valid object.
 *
 * There are cases where it's safe to use the naive code, but they're brittle.
 * For example, if `op` points to a Mechanica integer, you know that destroying
 * one of those can't cause problems -- but in part that relies on that
 * Mechanica integers aren't currently weakly referencable.  Best practice is
 * to use Mx_CLEAR() even if you can't think of a reason for why you need to.
 */
#define Mx_CLEAR(op)                            \
    do {                                        \
        MxObject *_py_tmp = (MxObject *)(op);   \
        if (_py_tmp != NULL) {                  \
            (op) = NULL;                        \
            Mx_DECREF(_py_tmp);                 \
        }                                       \
    } while (0)

/* Macros to use in case the object pointer may be NULL: */
#define Mx_XINCREF(op)                                \
    do {                                              \
        MxObject *_py_xincref_tmp = (MxObject *)(op); \
        if (_py_xincref_tmp != NULL)                  \
            Mx_INCREF(_py_xincref_tmp);               \
    } while (0)

#define Mx_XDECREF(op)                                \
    do {                                              \
        MxObject *_py_xdecref_tmp = (MxObject *)(op); \
        if (_py_xdecref_tmp != NULL)                  \
            Mx_DECREF(_py_xdecref_tmp);               \
    } while (0)

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Print an object, o, on file, fp.  Returns -1 on
 * error.  The flags argument is used to enable certain printing
 * options. The only option currently supported is Mx_Print_RAW.
 */
MxAPI_FUNC(int) MxObject_Print(MxObject *o, FILE *fp, int flags);

/**
 * Returns 1 if o has the attribute attr_name, and 0 otherwise.
 * This is equivalent to the Mechanica expression:
 * hasattr(o,attr_name).
 *
 * This function always succeeds.
 */
MxAPI_FUNC(int) MxObject_HasAttrString(MxObject *o, const char *attr_name);

/**
 * Retrieve an attributed named attr_name form object o.
 * Returns the attribute value on success, or NULL on failure.
 * This is the equivalent of the Mechanica expression: o.attr_name.
 */
MxAPI_FUNC(MxObject)* MxObject_GetAttrString(MxObject *o,
		const char *attr_name);


/**
 * Returns 1 if o has the attribute attr_name, and 0 otherwise.
 * This is equivalent to the Mechanica expression:
 * hasattr(o,attr_name).
 * This function always succeeds.
 */
MxAPI_FUNC(int) MxObject_HasAttr(MxObject *o, MxObject *attr_name);

/**
 * Retrieve an attributed named attr_name form object o.
 * Returns the attribute value on success, or NULL on failure.
 * This is the equivalent of the Mechanica expression: o.attr_name.
 */
MxAPI_FUNC(MxObject)* MxObject_GetAttr(MxObject *o, MxObject *attr_name);

/**
 * Set the value of the attribute named attr_name, for object o,
 * to the value, v. Returns -1 on failure.  This is
 * the equivalent of the Mechanica statement: o.attr_name=v.
 */
MxAPI_FUNC(int) MxObject_SetAttrString(MxObject *o, const char *attr_name,
		MxObject *v);

/**
 * Set the value of the attribute named attr_name, for object o,
 * to the value, v. Returns -1 on failure.  This is
 * the equivalent of the Mechanica statement: o.attr_name=v.
 */
MxAPI_FUNC(int) MxObject_SetAttr(MxObject *o, MxObject *attr_name, MxObject *v);

/**
 * Delete attribute named attr_name, for object o. Returns
 * -1 on failure.  This is the equivalent of the Mechanica
 * statement: del o.attr_name.
 */
MxAPI_FUNC(int) MxObject_DelAttrString(MxObject *o, const char *attr_name);

/**
 * Delete attribute named attr_name, for object o. Returns -1
 * on failure.  This is the equivalent of the Mechanica
 * statement: del o.attr_name.
 */
MxAPI_FUNC(int) MxObject_DelAttr(MxObject *o, MxObject *attr_name);

/**
 * Compute the string representation of object, o.  Returns the
 * string representation on success, NULL on failure.  This is
 * the equivalent of the Mechanica expression: repr(o).
 *
 * Mxlled by the repr() built-in function.
 */
MxAPI_FUNC(MxObject) *MxObject_Repr(MxObject *o);

/**
 * Compute the string representation of object, o.  Returns the
 * string representation on success, NULL on failure.  This is
 * the equivalent of the Mechanica expression: str(o).)
 *
 * Mxlled by the str() and print() built-in functions.
 */
MxAPI_FUNC(MxObject) *MxObject_Str(MxObject *o);

/**
 * Mxll the method named m of object o with a variable number of
 * C arguments. Returns the result of the call
 * on success, or NULL on failure.  This is the equivalent of
 * the Mechanica expression: o.method(args).
 */
MxAPI_FUNC(MxObject *) MxObject_CallMethod(MxObject *o,
		const char *method,
		const char *format, ...);

/**
 * Mxll the method named m of object o with a variable number of
 * C arguments.  The C arguments are provided as MxObject *
 * values, terminated by NULL.  Returns the result of the call
 * on success, or NULL on failure.  This is the equivalent of
 * the Mechanica expression: o.method(args).
 */
MxAPI_FUNC(MxObject *) MxObject_CallMethodObjArgs(MxObject *o,
		MxObject *method, ...);

/**
 *  Compute and return the hash, hash_value, of an object, o.  On
 failure, return -1.  This is the equivalent of the Mechanica
 expression: hash(o).
 */

MxAPI_FUNC(long) MxObject_Hash(MxObject *o);

/**
 *  Returns 1 if the object, o, is considered to be true, 0 if o is
 considered to be false and -1 on failure. This is equivalent to the
 Mechanica expression: not not o
 */

MxAPI_FUNC(int) MxObject_IsTrue(MxObject *o);

/**
 * Returns 0 if the object, o, is considered to be true, 1 if o is
 considered to be false and -1 on failure. This is equivalent to the
 Mechanica expression: not o
 */
MxAPI_FUNC(int) MxObject_Not(MxObject *o);

/**
 * On success, returns a type object corresponding to the object
 * type of object o. On failure, returns NULL.  This is
 * equivalent to the Mechanica expression: type(o).
 */
MxAPI_FUNC(MxObject *) MxObject_Type(MxObject *o);

/**
 * Return the size of object o.  If the object, o, provides
 * both sequence and mapping protocols, the sequence size is
 * returned. On error, -1 is returned.  This is the equivalent
 * to the Mechanica expression: len(o).
 */
MxAPI_FUNC(Mx_ssize_t) MxObject_Size(MxObject *o);

/**
 * Guess the size of object o using len(o) or o.__length_hint__().
 * If neither of those return a non-negative value, then return the
 * default value.  If one of the calls fails, this function returns -1.
 */
MxAPI_FUNC(Mx_ssize_t) MxObject_Length(MxObject *o);

/**
 * Return element of o corresponding to the object, key, or NULL
 * on failure. This is the equivalent of the Mechanica expression:
 * o[key].
 */
MxAPI_FUNC(MxObject *) MxObject_GetItem(MxObject *o, MxObject *key);

/**
 * Map the object, key, to the value, v.  Returns
 * -1 on failure.  This is the equivalent of the Mechanica
 * statement: o[key]=v.
 */
MxAPI_FUNC(int) MxObject_SetItem(MxObject *o, MxObject *key, MxObject *v);

/**
 * Remove the mapping for object, key, from the object *o.
 Returns -1 on failure.  This is equivalent to
 the Mechanica statement: del o[key].
 */

MxAPI_FUNC(int) MxObject_DelItemString(MxObject *o, const char *key);

/**
 * Delete the mapping for key from *o.  Returns -1 on failure.
 * This is the equivalent of the Mechanica statement: del o[key].
 */
MxAPI_FUNC(int) MxObject_DelItem(MxObject *o, MxObject *key);


MxAPI_FUNC(int) MxObject_IsInstance(MxObject *object, MxObject *typeorclass);
/* isinstance(object, typeorclass) */

MxAPI_FUNC(int) MxObject_IsSubclass(MxObject *object, MxObject *typeorclass);
/* issubclass(object, typeorclass) */

MxAPI_FUNC(void) Mx_Dealloc(MxObject *);

MxAPI_FUNC(uint32_t) Mx_IncRef(MxObject *o);

MxAPI_FUNC(uint32_t) Mx_DecRef(MxObject *o);




#ifdef __cplusplus
}
#endif

#endif /* _INCLUDED_CA_OBJECT_H_ */
