/*
 * MxType.h
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#ifndef SRC_MXTYPE_H_
#define SRC_MXTYPE_H_

#include "Mechanica.h"

#include "MxObject.h"

struct MxType : _typeobject{};



/* Object and type object interface */

/*
Objects are structures allocated on the heap.  Special rules apply to
the use of objects to ensure they are properly garbage-collected.
Objects are never allocated statically or on the stack; they must be
accessed through special macros and functions only.  (Type objects are
exceptions to the first rule; the standard types are represented by
statically initialized type objects, although work on type/class unification
for Mxthon 2.2 made it possible to have heap-allocated type objects too).

An object has a 'reference count' that is increased or decreased when a
pointer to the object is copied or deleted; when the reference count
reaches zero there are no references to the object left and it can be
removed from the heap.

An object has a 'type' that determines what it represents and what kind
of data it contains.  An object's type is fixed when it is created.
Types themselves are represented as objects; an object contains a
pointer to the corresponding type object.  The type itself has a type
pointer pointing to the object representing the type 'type', which
contains a pointer to itself!).

Objects do not float around in memory; once allocated an object keeps
the same size and address.  Objects that must hold variable-size data
can contain pointers to variable-size parts of the object.  Not all
objects of the same type have the same size; but the size cannot change
after allocation.  (These restrictions are made so a reference to an
object can be simply a pointer -- moving an object would require
updating all the pointers, and changing an object's size would require
moving it if there was another object right next to it.)

Objects are always accessed through pointers of the type 'MxObject *'.
The type 'MxObject' is a structure that only contains the reference count
and the type pointer.  The actual memory allocated for an object
contains other data that can only be accessed after casting the pointer
to a pointer to a longer structure type.  This longer type must start
with the reference count and type fields; the macro MxObject_HEAD should be
used for this (to accommodate for future changes).  The implementation
of a particular object type can cast the object pointer to the proper
type and back.

A standard interface exists for objects that contain an array of items
whose size is determined when the object is allocated.
*/

/* Mx_DEBUG implies Mx_TRACE_REFS. */
#if defined(Mx_DEBUG) && !defined(Mx_TRACE_REFS)
#define Mx_TRACE_REFS
#endif

/* Mx_TRACE_REFS implies Mx_REF_DEBUG. */
#if defined(Mx_TRACE_REFS) && !defined(Mx_REF_DEBUG)
#define Mx_REF_DEBUG
#endif

#ifdef Mx_TRACE_REFS
/* Define pointers to support a doubly-linked list of all live heap objects. */
#define _MxObject_HEAD_EXTRA            \
    struct _object *_ob_next;           \
    struct _object *_ob_prev;

#define _MxObject_EXTRA_INIT 0, 0,

#else
#define _MxObject_HEAD_EXTRA
#define _MxObject_EXTRA_INIT
#endif

/* MxObject_HEAD defines the initial segment of every MxObject. */
#define MxObject_HEAD                   \
    _MxObject_HEAD_EXTRA                \
    Mx_ssize_t ob_refcnt;               \
    struct _typeobject *ob_type;

#define MxObject_HEAD_INIT(type)        \
    _MxObject_EXTRA_INIT                \
    1, type,

#define MxVarObject_HEAD_INIT(type, size)       \
    MxObject_HEAD_INIT(type) size,

/* MxObject_VAR_HEAD defines the initial segment of all variable-size
 * container objects.  These end with a declaration of an array with 1
 * element, but enough space is malloc'ed so that the array actually
 * has room for ob_size elements.  Note that ob_size is an element count,
 * not necessarily a byte count.
 */
#define MxObject_VAR_HEAD               \
    MxObject_HEAD                       \
    Mx_ssize_t ob_size; /* Number of items in variable part */
#define Mx_INVALID_SIZE (Mx_ssize_t)-1

/* Nothing is actually declared to be a MxObject, but every pointer to
 * a Mxthon object can be cast to a MxObject*.  This is inheritance built
 * by hand.  Similarly every pointer to a variable-size Mxthon object can,
 * in addition, be cast to MxVarObject*.
 */
//typedef struct _object {
//   MxObject_HEAD
//} MxObject;

typedef struct {
    MxObject_VAR_HEAD
} MxVarObject;

#define Mx_REFCNT(ob)           (((MxObject*)(ob))->ob_refcnt)
#define Mx_TYPE(ob)             (((MxObject*)(ob))->ob_type)
#define Mx_SIZE(ob)             (((PyVarObject*)(ob))->ob_size)

/*
Type objects contain a string containing the type name (to help somewhat
in debugging), the allocation parameters (see MxObject_New() and
MxObject_NewVar()),
and methods for accessing objects of the type.  Methods are optional, a
nil pointer meaning that particular kind of access is not available for
this type.  The Mx_DECREF() macro uses the tp_dealloc method without
checking for a nil pointer; it should always be implemented except if
the implementation can guarantee that the reference count will never
reach zero (e.g., for statically allocated type objects).

NB: the methods for certain type groups are now contained in separate
method blocks.
*/




    /* Flags for getting buffers */
#define MxBUF_SIMPLE 0
#define MxBUF_WRITABLE 0x0001
/*  we used to include an E, backwards compatible alias  */
#define MxBUF_WRITEABLE MxBUF_WRITABLE
#define MxBUF_FORMAT 0x0004
#define MxBUF_ND 0x0008
#define MxBUF_STRIDES (0x0010 | MxBUF_ND)
#define MxBUF_C_CONTIGUOUS (0x0020 | MxBUF_STRIDES)
#define MxBUF_F_CONTIGUOUS (0x0040 | MxBUF_STRIDES)
#define MxBUF_ANY_CONTIGUOUS (0x0080 | MxBUF_STRIDES)
#define MxBUF_INDIRECT (0x0100 | MxBUF_STRIDES)

#define MxBUF_CONTIG (MxBUF_ND | MxBUF_WRITABLE)
#define MxBUF_CONTIG_RO (MxBUF_ND)

#define MxBUF_STRIDED (MxBUF_STRIDES | MxBUF_WRITABLE)
#define MxBUF_STRIDED_RO (MxBUF_STRIDES)

#define MxBUF_RECORDS (MxBUF_STRIDES | MxBUF_WRITABLE | MxBUF_FORMAT)
#define MxBUF_RECORDS_RO (MxBUF_STRIDES | MxBUF_FORMAT)

#define MxBUF_FULL (MxBUF_INDIRECT | MxBUF_WRITABLE | MxBUF_FORMAT)
#define MxBUF_FULL_RO (MxBUF_INDIRECT | MxBUF_FORMAT)


#define MxBUF_READ  0x100
#define MxBUF_WRITE 0x200
#define MxBUF_SHADOW 0x400
/* end Mx3k buffer interface */



/* access macro to the members which are floating "behind" the object */
#define MxHeapType_GET_MEMBERS(etype) \
    ((PyMemberDef *)(((char *)etype) + Mx_TYPE(etype)->tp_basicsize))


/* Generic type check */
MxAPI_FUNC(int) MxType_IsSubtype(MxType *, MxType *);
#define MxObject_TypeCheck(ob, tp) \
    (Mx_TYPE(ob) == (tp) || MxType_IsSubtype(Mx_TYPE(ob), (tp)))

MxAPI_DATA(MxType) MxType_Type; /* built-in 'type' */
MxAPI_DATA(MxType) MxBaseObject_Type; /* built-in 'object' */
MxAPI_DATA(MxType) MxSuper_Type; /* built-in 'super' */

#define MxType_Check(op) \
    MxType_FastSubclass(Mx_TYPE(op), Mx_TPFLAGS_TYPE_SUBCLASS)
#define MxType_CheckExact(op) (Mx_TYPE(op) == &PyType_Type)

MxAPI_FUNC(int) MxType_Ready(MxType *);
MxAPI_FUNC(MxObject *) MxType_GenericAlloc(MxType *, Mx_ssize_t);
MxAPI_FUNC(MxObject *) MxType_GenericNew(MxType *,
                                               MxObject *, MxObject *);
MxAPI_FUNC(MxObject *) _MxType_Lookup(MxType *, MxObject *);
MxAPI_FUNC(MxObject *) _MxObject_LookupSpecial(MxObject *, char *, MxObject **);
MxAPI_FUNC(unsigned int) MxType_ClearCache(void);
MxAPI_FUNC(void) MxType_Modified(MxType *);

/* Generic operations on objects */
MxAPI_FUNC(int) MxObject_Print(MxObject *, FILE *, int);
MxAPI_FUNC(void) _MxObject_Dump(MxObject *);
MxAPI_FUNC(MxObject *) MxObject_Repr(MxObject *);
MxAPI_FUNC(MxObject *) _MxObject_Str(MxObject *);
MxAPI_FUNC(MxObject *) MxObject_Str(MxObject *);
#define MxObject_Bytes MxObject_Str
#ifdef Mx_USING_UNICODE
MxAPI_FUNC(MxObject *) MxObject_Unicode(MxObject *);
#endif
MxAPI_FUNC(int) MxObject_Compare(MxObject *, MxObject *);
MxAPI_FUNC(MxObject *) MxObject_RichCompare(MxObject *, MxObject *, int);
MxAPI_FUNC(int) MxObject_RichCompareBool(MxObject *, MxObject *, int);
MxAPI_FUNC(MxObject *) MxObject_GetAttrString(MxObject *, const char *);
MxAPI_FUNC(int) MxObject_SetAttrString(MxObject *, const char *, MxObject *);
MxAPI_FUNC(int) MxObject_HasAttrString(MxObject *, const char *);
MxAPI_FUNC(MxObject *) MxObject_GetAttr(MxObject *, MxObject *);
MxAPI_FUNC(int) MxObject_SetAttr(MxObject *, MxObject *, MxObject *);
MxAPI_FUNC(int) MxObject_HasAttr(MxObject *, MxObject *);
MxAPI_FUNC(MxObject **) _MxObject_GetDictPtr(MxObject *);
MxAPI_FUNC(MxObject *) MxObject_SelfIter(MxObject *);
MxAPI_FUNC(MxObject *) _MxObject_NextNotImplemented(MxObject *);
MxAPI_FUNC(MxObject *) MxObject_GenericGetAttr(MxObject *, MxObject *);
MxAPI_FUNC(int) MxObject_GenericSetAttr(MxObject *,
                                              MxObject *, MxObject *);
MxAPI_FUNC(long) MxObject_Hash(MxObject *);
MxAPI_FUNC(long) MxObject_HashNotImplemented(MxObject *);
MxAPI_FUNC(int) MxObject_IsTrue(MxObject *);
MxAPI_FUNC(int) MxObject_Not(MxObject *);
MxAPI_FUNC(int) MxCallable_Check(MxObject *);
MxAPI_FUNC(int) MxNumber_Coerce(MxObject **, MxObject **);
MxAPI_FUNC(int) MxNumber_CoerceEx(MxObject **, MxObject **);

MxAPI_FUNC(void) MxObject_ClearWeakRefs(MxObject *);

/* A slot function whose address we need to compare */
extern int _MxObject_SlotCompare(MxObject *, MxObject *);
/* Same as MxObject_Generic{Get,Set}Attr, but passing the attributes
   dict as the last parameter. */
MxAPI_FUNC(MxObject *)
_MxObject_GenericGetAttrWithDict(MxObject *, MxObject *, MxObject *);
MxAPI_FUNC(int)
_MxObject_GenericSetAttrWithDict(MxObject *, MxObject *,
                                 MxObject *, MxObject *);


/* MxObject_Dir(obj) acts like Mxthon __builtin__.dir(obj), returning a
   list of strings.  MxObject_Dir(NULL) is like __builtin__.dir(),
   returning the names of the current locals.  In this case, if there are
   no current locals, NULL is returned, and MxErr_Occurred() is false.
*/
MxAPI_FUNC(MxObject *) MxObject_Dir(MxObject *);


/* Helpers for printing recursive container types */
MxAPI_FUNC(int) Mx_ReprEnter(MxObject *);
MxAPI_FUNC(void) Mx_ReprLeave(MxObject *);

/* Helpers for hash functions */
MxAPI_FUNC(long) _Mx_HashDouble(double);
MxAPI_FUNC(long) _Mx_HashPointer(void*);

typedef struct {
    long prefix;
    long suffix;
} _Mx_HashSecret_t;
MxAPI_DATA(_Mx_HashSecret_t) _Mx_HashSecret;

#ifdef Mx_DEBUG
MxAPI_DATA(int) _Mx_HashSecret_Initialized;
#endif

/* Helper for passing objects to printf and the like.
   Leaks refcounts.  Don't use it!
*/
#define MxObject_REPR(obj) MxString_AS_STRING(MxObject_Repr(obj))

/* Flag bits for printing: */
#define Mx_PRINT_RAW    1       /* No string quotes etc. */

/*
`Type flags (tp_flags)

These flags are used to extend the type structure in a backwards-compatible
fashion. Extensions can use the flags to indicate (and test) when a given
type structure contains a new feature. The Mxthon core will use these when
introducing new functionality between major revisions (to avoid mid-version
changes in the PYTHON_API_VERSION).

Arbitration of the flag bit positions will need to be coordinated among
all extension writers who publically release their extensions (this will
be fewer than you might expect!)..

Python 1.5.2 introduced the bf_getcharbuffer slot into MxBufferProcs.

Type definitions should use Mx_TPFLAGS_DEFAULT for their tp_flags value.

Code can use MxType_HasFeature(type_ob, flag_value) to test whether the
given type object has a specified feature.

NOTE: when building the core, Mx_TPFLAGS_DEFAULT includes
Mx_TPFLAGS_HAVE_VERSION_TAG; outside the core, it doesn't.  This is so
that extensions that modify tp_dict of their own types directly don't
break, since this was allowed in 2.5.  In 3.0 they will have to
manually remove this flag though!
*/

/* MxBufferProcs contains bf_getcharbuffer */
#define Mx_TPFLAGS_HAVE_GETCHARBUFFER  (1L<<0)

/* MxSequenceMethods contains sq_contains */
#define Mx_TPFLAGS_HAVE_SEQUENCE_IN (1L<<1)

/* This is here for backwards compatibility.  Extensions that use the old GC
 * API will still compile but the objects will not be tracked by the GC. */
#define Mx_TPFLAGS_GC 0 /* used to be (1L<<2) */

/* MxSequenceMethods and MxNumberMethods contain in-place operators */
#define Mx_TPFLAGS_HAVE_INPLACEOPS (1L<<3)

/* MxNumberMethods do their own coercion */
#define Mx_TPFLAGS_CHECKTYPES (1L<<4)

/* tp_richcompare is defined */
#define Mx_TPFLAGS_HAVE_RICHCOMPARE (1L<<5)

/* Objects which are weakly referencable if their tp_weaklistoffset is >0 */
#define Mx_TPFLAGS_HAVE_WEAKREFS (1L<<6)

/* tp_iter is defined */
#define Mx_TPFLAGS_HAVE_ITER (1L<<7)

/* New members introduced by Mxthon 2.2 exist */
#define Mx_TPFLAGS_HAVE_CLASS (1L<<8)

/* Set if the type object is dynamically allocated */
#define Mx_TPFLAGS_HEAPTYPE (1L<<9)

/* Set if the type allows subclassing */
#define Mx_TPFLAGS_BASETYPE (1L<<10)

/* Set if the type is 'ready' -- fully initialized */
#define Mx_TPFLAGS_READY (1L<<12)

/* Set while the type is being 'readied', to prevent recursive ready calls */
#define Mx_TPFLAGS_READYING (1L<<13)

/* Objects support garbage collection (see objimp.h) */
#define Mx_TPFLAGS_HAVE_GC (1L<<14)

/* These two bits are preserved for Stackless Mxthon, next after this is 17 */
#ifdef STACKLESS
#define Mx_TPFLAGS_HAVE_STACKLESS_EXTENSION (3L<<15)
#else
#define Mx_TPFLAGS_HAVE_STACKLESS_EXTENSION 0
#endif

/* Objects support nb_index in MxNumberMethods */
#define Mx_TPFLAGS_HAVE_INDEX (1L<<17)

/* Objects support type attribute cache */
#define Mx_TPFLAGS_HAVE_VERSION_TAG   (1L<<18)
#define Mx_TPFLAGS_VALID_VERSION_TAG  (1L<<19)

/* Type is abstract and cannot be instantiated */
#define Mx_TPFLAGS_IS_ABSTRACT (1L<<20)

/* Has the new buffer protocol */
#define Mx_TPFLAGS_HAVE_NEWBUFFER (1L<<21)

/* These flags are used to determine if a type is a subclass. */
#define Mx_TPFLAGS_INT_SUBCLASS         (1L<<23)
#define Mx_TPFLAGS_LONG_SUBCLASS        (1L<<24)
#define Mx_TPFLAGS_LIST_SUBCLASS        (1L<<25)
#define Mx_TPFLAGS_TUPLE_SUBCLASS       (1L<<26)
#define Mx_TPFLAGS_STRING_SUBCLASS      (1L<<27)
#define Mx_TPFLAGS_UNICODE_SUBCLASS     (1L<<28)
#define Mx_TPFLAGS_DICT_SUBCLASS        (1L<<29)
#define Mx_TPFLAGS_BASE_EXC_SUBCLASS    (1L<<30)
#define Mx_TPFLAGS_TYPE_SUBCLASS        (1L<<31)

#define Mx_TPFLAGS_DEFAULT_EXTERNAL ( \
                 Mx_TPFLAGS_HAVE_GETCHARBUFFER | \
                 Mx_TPFLAGS_HAVE_SEQUENCE_IN | \
                 Mx_TPFLAGS_HAVE_INPLACEOPS | \
                 Mx_TPFLAGS_HAVE_RICHCOMPARE | \
                 Mx_TPFLAGS_HAVE_WEAKREFS | \
                 Mx_TPFLAGS_HAVE_ITER | \
                 Mx_TPFLAGS_HAVE_CLASS | \
                 Mx_TPFLAGS_HAVE_STACKLESS_EXTENSION | \
                 Mx_TPFLAGS_HAVE_INDEX | \
                 0)
#define Mx_TPFLAGS_DEFAULT_CORE (Mx_TPFLAGS_DEFAULT_EXTERNAL | \
                 Mx_TPFLAGS_HAVE_VERSION_TAG)

#ifdef Mx_BUILD_CORE
#define Mx_TPFLAGS_DEFAULT Mx_TPFLAGS_DEFAULT_CORE
#else
#define Mx_TPFLAGS_DEFAULT Mx_TPFLAGS_DEFAULT_EXTERNAL
#endif

#define MxType_HasFeature(t,f)  (((t)->tp_flags & (f)) != 0)
#define MxType_FastSubclass(t,f)  MxType_HasFeature(t,f)


/*
The macros Mx_INCREF(op) and Mx_DECREF(op) are used to increment or decrement
reference counts.  Mx_DECREF calls the object's deallocator function when
the refcount falls to 0; for
objects that don't contain references to other objects or heap memory
this can be the standard function free().  Both macros can be used
wherever a void expression is allowed.  The argument must not be a
NULL pointer.  If it may be NULL, use Mx_XINCREF/Mx_XDECREF instead.
The macro _Mx_NewReference(op) initialize reference counts to 1, and
in special builds (Mx_REF_DEBUG, Mx_TRACE_REFS) performs additional
bookkeeping appropriate to the special build.

We assume that the reference count field can never overflow; this can
be proven when the size of the field is the same as the pointer size, so
we ignore the possibility.  Provided a C int is at least 32 bits (which
is implicitly assumed in many parts of this code), that's enough for
about 2**31 references to an object.

XXX The following became out of date in Mxthon 2.2, but I'm not sure
XXX what the full truth is now.  Certainly, heap-allocated type objects
XXX can and should be deallocated.
Type objects should never be deallocated; the type pointer in an object
is not considered to be a reference to the type object, to save
complications in the deallocation function.  (This is actually a
decision that's up to the implementer of each new type so if you want,
you can count such references to the type object.)

*** WARNING*** The Mx_DECREF macro must have a side-effect-free argument
since it may evaluate its argument multiple times.  (The alternative
would be to mace it a proper function or assign it to a global temporary
variable first, both of which are slower; and in a multi-threaded
environment the global variable trick is not safe.)
*/

/* First define a pile of simple helper macros, one set per special
 * build symbol.  These either expand to the obvious things, or to
 * nothing at all when the special mode isn't in effect.  The main
 * macros can later be defined just once then, yet expand to different
 * things depending on which special build options are and aren't in effect.
 * Trust me <wink>:  while painful, this is 20x easier to understand than,
 * e.g, defining _Mx_NewReference five different times in a maze of nested
 * #ifdefs (we used to do that -- it was impenetrable).
 */
#ifdef Mx_REF_DEBUG
MxAPI_DATA(Mx_ssize_t) _Mx_RefTotal;
MxAPI_FUNC(void) _Mx_NegativeRefcount(const char *fname,
                                            int lineno, MxObject *op);
MxAPI_FUNC(MxObject *) _PyDict_Dummy(void);
MxAPI_FUNC(MxObject *) _PySet_Dummy(void);
MxAPI_FUNC(Mx_ssize_t) _Mx_GetRefTotal(void);
#define _Mx_INC_REFTOTAL        _Mx_RefTotal++
#define _Mx_DEC_REFTOTAL        _Mx_RefTotal--
#define _Mx_REF_DEBUG_COMMA     ,
#define _Mx_CHECK_REFCNT(OP)                                    \
{       if (((MxObject*)OP)->ob_refcnt < 0)                             \
                _Mx_NegativeRefcount(__FILE__, __LINE__,        \
                                     (MxObject *)(OP));         \
}
#else
#define _Mx_INC_REFTOTAL
#define _Mx_DEC_REFTOTAL
#define _Mx_REF_DEBUG_COMMA
#define _Mx_CHECK_REFCNT(OP)    /* a semicolon */;
#endif /* Mx_REF_DEBUG */

#ifdef COUNT_ALLOCS
MxAPI_FUNC(void) inc_count(MxType *);
MxAPI_FUNC(void) dec_count(MxType *);
#define _Mx_INC_TPALLOCS(OP)    inc_count(Mx_TYPE(OP))
#define _Mx_INC_TPFREES(OP)     dec_count(Mx_TYPE(OP))
#define _Mx_DEC_TPFREES(OP)     Mx_TYPE(OP)->tp_frees--
#define _Mx_COUNT_ALLOCS_COMMA  ,
#else
#define _Mx_INC_TPALLOCS(OP)
#define _Mx_INC_TPFREES(OP)
#define _Mx_DEC_TPFREES(OP)
#define _Mx_COUNT_ALLOCS_COMMA
#endif /* COUNT_ALLOCS */

#ifdef Mx_TRACE_REFS
/* Mx_TRACE_REFS is such major surgery that we call external routines. */
MxAPI_FUNC(void) _Mx_NewReference(MxObject *);
MxAPI_FUNC(void) _Mx_ForgetReference(MxObject *);
MxAPI_FUNC(void) _Mx_Dealloc(MxObject *);
MxAPI_FUNC(void) _Mx_PrintReferences(FILE *);
MxAPI_FUNC(void) _Mx_PrintReferenceAddresses(FILE *);
MxAPI_FUNC(void) _Mx_AddToAllObjects(MxObject *, int force);

#else
/* Without Mx_TRACE_REFS, there's little enough to do that we expand code
 * inline.
 */
#define _Mx_NewReference(op) (                          \
    _Mx_INC_TPALLOCS(op) _Mx_COUNT_ALLOCS_COMMA         \
    _Mx_INC_REFTOTAL  _Mx_REF_DEBUG_COMMA               \
    Mx_REFCNT(op) = 1)

#define _Mx_ForgetReference(op) _Mx_INC_TPFREES(op)

#define _Mx_Dealloc(op) (                               \
    _Mx_INC_TPFREES(op) _Mx_COUNT_ALLOCS_COMMA          \
    (*Mx_TYPE(op)->tp_dealloc)((MxObject *)(op)))
#endif /* !Mx_TRACE_REFS */

#define Mx_INCREF(op) (                         \
    _Mx_INC_REFTOTAL  _Mx_REF_DEBUG_COMMA       \
    ((MxObject*)(op))->ob_refcnt++)

#define Mx_DECREF(op)                                   \
    do {                                                \
        if (_Mx_DEC_REFTOTAL  _Mx_REF_DEBUG_COMMA       \
        --((MxObject*)(op))->ob_refcnt != 0)            \
            _Mx_CHECK_REFCNT(op)                        \
        else                                            \
        _Mx_Dealloc((MxObject *)(op));                  \
    } while (0)

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
 * coded in Mxthon, which in turn can release the GIL and allow other threads
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
 * For example, if `op` points to a Mxthon integer, you know that destroying
 * one of those can't cause problems -- but in part that relies on that
 * Mxthon integers aren't currently weakly referencable.  Best practice is
 * to use Mx_CLEAR() even if you can't think of a reason for why you need to.
 */
//#define Mx_CLEAR(op)                            \
//    do {                                        \
//        if (op) {                               \
//            MxObject *_py_tmp = (MxObject *)(op);               \
//            (op) = NULL;                        \
//            Mx_DECREF(_py_tmp);                 \
//        }                                       \
//    } while (0)
//
///* Macros to use in case the object pointer may be NULL: */
//#define Mx_XINCREF(op) do { if ((op) == NULL) ; else Mx_INCREF(op); } while (0)
//#define Mx_XDECREF(op) do { if ((op) == NULL) ; else Mx_DECREF(op); } while (0)

/* Safely decref `op` and set `op` to `op2`.
 *
 * As in case of Mx_CLEAR "the obvious" code can be deadly:
 *
 *     Mx_DECREF(op);
 *     op = op2;
 *
 * The safe way is:
 *
 *      Mx_SETREF(op, op2);
 *
 * That arranges to set `op` to `op2` _before_ decref'ing, so that any code
 * triggered as a side-effect of `op` getting torn down no longer believes
 * `op` points to a valid object.
 *
 * Mx_XSETREF is a variant of Mx_SETREF that uses Mx_XDECREF instead of
 * Mx_DECREF.
 */

#define Mx_SETREF(op, op2)                      \
    do {                                        \
        MxObject *_py_tmp = (MxObject *)(op);   \
        (op) = (op2);                           \
        Mx_DECREF(_py_tmp);                     \
    } while (0)

#define Mx_XSETREF(op, op2)                     \
    do {                                        \
        MxObject *_py_tmp = (MxObject *)(op);   \
        (op) = (op2);                           \
        Mx_XDECREF(_py_tmp);                    \
    } while (0)

/*
These are provided as conveniences to Mxthon runtime embedders, so that
they can have object code that is not dependent on Mxthon compilation flags.
*/
//MxAPI_FUNC(void) Mx_IncRef(MxObject *);
//MxAPI_FUNC(void) Mx_DecRef(MxObject *);

/*
_Mx_NoneStruct is an object of undefined type which can be used in contexts
where NULL (nil) is not suitable (since NULL often means 'error').

Don't forget to apply Mx_INCREF() when returning this value!!!
*/
MxAPI_DATA(MxObject) _Mx_NoneStruct; /* Don't use this directly */
#define Mx_None (&_Mx_NoneStruct)

/* Macro for returning Mx_None from a function */
#define Mx_RETURN_NONE return Mx_INCREF(Mx_None), Mx_None

/*
Mx_NotImplemented is a singleton used to signal that an operation is
not implemented for a given type combination.
*/
MxAPI_DATA(MxObject) _Mx_NotImplementedStruct; /* Don't use this directly */
#define Mx_NotImplemented (&_Mx_NotImplementedStruct)

/* Rich comparison opcodes */
#define Mx_LT 0
#define Mx_LE 1
#define Mx_EQ 2
#define Mx_NE 3
#define Mx_GT 4
#define Mx_GE 5

/* Maps Mx_LT to Mx_GT, ..., Mx_GE to Mx_LE.
 * Defined in object.c.
 */
MxAPI_DATA(int) _Mx_SwappedOp[];

/*
Define staticforward and statichere for source compatibility with old
C extensions.

The staticforward define was needed to support certain broken C
compilers (notably SCO ODT 3.0, perhaps early AIX as well) botched the
static keyword when it was used with a forward declaration of a static
initialized structure.  Standard C allows the forward declaration with
static, and we've decided to stop catering to broken C compilers.
(In fact, we expect that the compilers are all fixed eight years later.)
*/

#define staticforward static
#define statichere static


/*
More conventions
================

Argument Checking
-----------------

Functions that take objects as arguments normally don't check for nil
arguments, but they do check the type of the argument, and return an
error if the function doesn't apply to the type.

Failure Modes
-------------

Functions may fail for a variety of reasons, including running out of
memory.  This is communicated to the caller in two ways: an error string
is set (see errors.h), and the function result differs: functions that
normally return a pointer return NULL for failure, functions returning
an integer return -1 (which could be a legal return value too!), and
other functions return 0 for success and -1 for failure.
Callers should always check for errors before using the result.  If
an error was set, the caller must either explicitly clear it, or pass
the error on to its caller.

Reference Counts
----------------

It takes a while to get used to the proper usage of reference counts.

Functions that create an object set the reference count to 1; such new
objects must be stored somewhere or destroyed again with Mx_DECREF().
Some functions that 'store' objects, such as MxTuple_SetItem() and
PyList_SetItem(),
don't increment the reference count of the object, since the most
frequent use is to store a fresh object.  Functions that 'retrieve'
objects, such as MxTuple_GetItem() and MxDict_GetItemString(), also
don't increment
the reference count, since most frequently the object is only looked at
quickly.  Thus, to retrieve an object and store it again, the caller
must call Mx_INCREF() explicitly.

NOTE: functions that 'consume' a reference count, like
PyList_SetItem(), consume the reference even if the object wasn't
successfully stored, to simplify error handling.

It seems attractive to make other functions that take an object as
argument consume a reference count; however, this may quickly get
confusing (even the current practice is already confusing).  Consider
it carefully, it may save lots of calls to Mx_INCREF() and Mx_DECREF() at
times.
*/


/* Trashcan mechanism, thanks to Christian Tismer.

When deallocating a container object, it's possible to trigger an unbounded
chain of deallocations, as each Mx_DECREF in turn drops the refcount on "the
next" object in the chain to 0.  This can easily lead to stack faults, and
especially in threads (which typically have less stack space to work with).

A container object that participates in cyclic gc can avoid this by
bracketing the body of its tp_dealloc function with a pair of macros:

static void
mytype_dealloc(mytype *p)
{
    ... declarations go here ...

    MxObject_GC_UnTrack(p);        // must untrack first
    Mx_TRASHCAN_SAFE_BEGIN(p)
    ... The body of the deallocator goes here, including all calls ...
    ... to Mx_DECREF on contained objects.                         ...
    Mx_TRASHCAN_SAFE_END(p)
}

CAUTION:  Never return from the middle of the body!  If the body needs to
"get out early", put a label immediately before the Mx_TRASHCAN_SAFE_END
call, and goto it.  Else the call-depth counter (see below) will stay
above 0 forever, and the trashcan will never get emptied.

How it works:  The BEGIN macro increments a call-depth counter.  So long
as this counter is small, the body of the deallocator is run directly without
further ado.  But if the counter gets large, it instead adds p to a list of
objects to be deallocated later, skips the body of the deallocator, and
resumes execution after the END macro.  The tp_dealloc routine then returns
without deallocating anything (and so unbounded call-stack depth is avoided).

When the call stack finishes unwinding again, code generated by the END macro
notices this, and calls another routine to deallocate all the objects that
may have been added to the list of deferred deallocations.  In effect, a
chain of N deallocations is broken into N / MxTrash_UNWIND_LEVEL pieces,
with the call stack never exceeding a depth of MxTrash_UNWIND_LEVEL.
*/




#define MxTrash_UNWIND_LEVEL 50

/* Note the workaround for when the thread state is NULL (issue #17703) */
#define Mx_TRASHCAN_SAFE_BEGIN(op) \
    do { \
        MxThreadState *_tstate = MxThreadState_GET(); \
        if (!_tstate || \
            _tstate->trash_delete_nesting < MxTrash_UNWIND_LEVEL) { \
            if (_tstate) \
                ++_tstate->trash_delete_nesting;
            /* The body of the deallocator is here. */
#define Mx_TRASHCAN_SAFE_END(op) \
            if (_tstate) { \
                --_tstate->trash_delete_nesting; \
                if (_tstate->trash_delete_later \
                    && _tstate->trash_delete_nesting <= 0) \
                    _PyTrash_thread_destroy_chain(); \
            } \
        } \
        else \
            _PyTrash_thread_deposit_object((MxObject*)op); \
    } while (0);


/**
 * Init and add to python module
 */
void MxType_init(PyObject *m);


#endif /* SRC_MXTYPE_H_ */
