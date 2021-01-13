/*
 * MxParticleList.cpp
 *
 *  Created on: Nov 23, 2020
 *      Author: andy
 */


#include <MxParticleList.hpp>
#include "engine.h"
#include <Magnum/Math/Distance.h>
#include <Magnum/Math/Matrix3.h>
#include <MxConvert.hpp>
#include <metrics.h>
#include <cstdarg>
#include <iostream>
#include <MxNumpy.h>

static PyObject* list_virial(MxParticleList *self, PyObject *args, PyObject *kwargs);

static PyObject* list_radius_of_gyration(MxParticleList *self, PyObject *args, PyObject *kwargs);

static PyObject* list_center_of_mass(MxParticleList *self, PyObject *args, PyObject *kwargs);

static PyObject* list_center_of_geometry(MxParticleList *self, PyObject *args, PyObject *kwargs);

static PyObject* list_moment_of_inertia(MxParticleList *self, PyObject *args, PyObject *kwargs);

static PyObject* list_copy(MxParticleList *self, PyObject *args, PyObject *kwargs);

static PyObject* list_positions(MxParticleList *self);

static PyObject* list_spherical_positions(MxParticleList *self, PyObject *args, PyObject *kwargs);



void MxParticleList::init()
{
    this->ob_type = &MxParticleList_Type;
    this->ob_refcnt = 1;
    this->parts = NULL;
    this->nr_parts = 0;
    this->size_parts = 0;
    this->flags = 0;
}

void MxParticleList::free()
{
}

uint16_t MxParticleList::insert(int32_t id)
{
    /* do we need to extend the partlist? */
    if ( nr_parts == size_parts ) {
        size_parts += space_partlist_incr;
        int32_t* temp = NULL;
        if (( temp = (int32_t*)malloc( sizeof(int32_t) * size_parts )) == NULL ) {
            return c_error(E_FAIL, "could not allocate space for type particles");
        }
        memcpy( temp , parts , sizeof(int32_t) * nr_parts );
        ::free( parts );
        parts = temp;
    }
    
    parts[nr_parts] = id;

    return nr_parts++;
}

uint16_t MxParticleList::remove(int32_t id)
{
    int i = 0;
    for(; i < nr_parts; i++) {
        if(parts[i] == id)
            break;
    }
    
    if(i == nr_parts) {
        return c_error(E_FAIL, "type does not contain particle id");
    }
    
    nr_parts--;
    if(i < nr_parts) {
        parts[i] = parts[nr_parts];
    }
    
    return i;
}

//typedef void (*freefunc)(void *);


/**
 * called when the new reference count is zero. At this point, the instance is
 * still in existence, but there are no references to it. The destructor
 * function should free all references which the instance owns, free all
 * memory buffers owned by the instance (using the freeing function corresponding
 * to the allocation function used to allocate the buffer), and call the
 * type’s tp_free function.
 */
void particlelist_dealloc(MxParticleList *p) {
    if(p->flags & PARTICLELIST_OWNDATA) {
        ::free(p->parts);
    }
    
    if(p->flags * PARTICLELIST_OWNSELF) {
        MxParticleList_Type.tp_free(p);
    }
}



// sq_length
static Py_ssize_t plist_length(PyObject *_self) {
    std::cout << MX_FUNCTION << std::endl;
    MxParticleList *self = (MxParticleList*)_self;
    return self->nr_parts;
}

// sq_concat
static PyObject *plist_concat(PyObject *, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_repeat
static PyObject *plist_repeat(PyObject *, Py_ssize_t) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_item
static PyObject *plist_item(PyObject *_self, Py_ssize_t i) {

    MxParticleList *self = (MxParticleList*)_self;
    
    if(i < self->nr_parts) {
        return _Engine.s.partlist[self->parts[i]]->py_particle();
    }
    else {
        PyErr_SetString(PyExc_IndexError, "cluster index out of range");
    }
    return NULL;
}

// sq_ass_item
static int plist_ass_item(PyObject *, Py_ssize_t, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_contains
static int plist_contains(PyObject *, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_inplace_concat
static PyObject *plist_inplace_concat(PyObject *, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_inplace_repeat
static PyObject *plist_inplace_repeat(PyObject *, Py_ssize_t) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}


static PySequenceMethods sequence_methods =  {
    plist_length, // lenfunc sq_length;
    plist_concat, // binaryfunc sq_concat;
    plist_repeat, // ssizeargfunc sq_repeat;
    plist_item, // ssizeargfunc sq_item;
    0, // void *was_sq_slice;
    plist_ass_item, // ssizeobjargproc sq_ass_item;
    0, // void *was_sq_ass_slice;
    plist_contains, // objobjproc sq_contains;
    plist_inplace_concat, // binaryfunc sq_inplace_concat;
    plist_inplace_repeat  // ssizeargfunc sq_inplace_repeat;
};

static PyMethodDef list_methods[] = {
    { "virial", (PyCFunction)list_virial, METH_VARARGS | METH_KEYWORDS, NULL },
    { "radius_of_gyration", (PyCFunction)list_radius_of_gyration, METH_VARARGS | METH_KEYWORDS, NULL },
    { "center_of_mass", (PyCFunction)list_center_of_mass, METH_VARARGS | METH_KEYWORDS, NULL },
    { "center_of_geometry", (PyCFunction)list_center_of_geometry, METH_VARARGS | METH_KEYWORDS, NULL },
    { "centroid", (PyCFunction)list_center_of_geometry, METH_VARARGS | METH_KEYWORDS, NULL },
    { "moment_of_inertia", (PyCFunction)list_moment_of_inertia, METH_VARARGS | METH_KEYWORDS, NULL },
    { "inertia", (PyCFunction)list_moment_of_inertia, METH_VARARGS | METH_KEYWORDS, NULL },
    { "copy", (PyCFunction)list_copy, METH_VARARGS | METH_KEYWORDS, NULL },
    { "positions", (PyCFunction)list_positions, METH_NOARGS , NULL },
    { "spherical_positions", (PyCFunction)list_spherical_positions, METH_VARARGS | METH_KEYWORDS, NULL },
    { NULL, NULL, 0, NULL }
};

PyTypeObject MxParticleList_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ParticleList",
    .tp_basicsize = sizeof(MxParticleList),
    .tp_itemsize =       0,
    .tp_dealloc =        (destructor)particlelist_dealloc,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    &sequence_methods,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            0,
    .tp_getattro =       0,
    .tp_setattro =       0,
    .tp_as_buffer =      0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "Custom objects",
    .tp_traverse =       0,
    .tp_clear =          0,
    .tp_richcompare =    0,
    .tp_weaklistoffset = 0,
    .tp_iter =           0,
    .tp_iternext =       0,
    .tp_methods =        list_methods,
    .tp_members =        0,
    .tp_getset =         0,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           0,
    .tp_alloc =          0,
    .tp_new =            0,
    .tp_free =           0,
    .tp_is_gc =          0,
    .tp_bases =          0,
    .tp_mro =            0,
    .tp_cache =          0,
    .tp_subclasses =     0,
    .tp_weaklist =       0,
    .tp_del =            0,
    .tp_version_tag =    0,
    .tp_finalize =       0,
};


HRESULT _MxParticleList_init(PyObject *m)
{
    if (PyType_Ready((PyTypeObject*)&MxParticleList_Type) < 0) {
        return E_FAIL;
    }
    
    Py_INCREF(&MxParticleList_Type);
    if (PyModule_AddObject(m, "ParticleList", (PyObject *)&MxParticleList_Type) < 0) {
        Py_DECREF(&MxParticleList_Type);
        return E_FAIL;
    }
    
    return S_OK;
}

MxParticleList* MxParticleList_New(uint16_t init_size,
        uint16_t flags)
{
    MxParticleList *list = (MxParticleList*)PyType_GenericAlloc(&MxParticleList_Type, 0);
    list->flags = flags;
    list->size_parts = init_size;
    list->parts = (int32_t*)malloc(init_size * sizeof(int32_t));
    list->nr_parts = 0;
    
    // we allocated usign Python, so this object both owns it's own memory and the
    // data memory
    list->flags |= PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF;
    
    return list;
}

MxParticleList* MxParticleList_NewFromData(uint16_t nr_parts, int32_t *parts) {
    MxParticleList *list = (MxParticleList*)PyType_GenericAlloc(&MxParticleList_Type, 0);
    list->flags = PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF;
    list->size_parts = nr_parts;
    list->parts = parts;
    list->nr_parts = nr_parts;
    return list;
}

int MxParticleList_Check(const PyObject *obj) {
    return PyObject_IsInstance(const_cast<PyObject*>(obj), (PyObject*)&MxParticleList_Type);
}

static MxParticleList* particlelist_from_list(PyObject *list) {
    int nr_parts = PyList_Size(list);
    
    MxParticleList* pl = (MxParticleList*)PyType_GenericNew(&MxParticleList_Type, NULL, NULL);
    pl->flags = PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF;
    pl->size_parts = nr_parts;
    pl->parts = (int32_t*)malloc(nr_parts * sizeof(int32_t));;
    pl->nr_parts = nr_parts;
    
    for(int i = 0; i < nr_parts; ++i) {
        PyObject *obj = PyList_GET_ITEM(list, i);
        MxParticle *p = MxParticle_Get(obj);
        if(!p) {
            Py_DECREF(pl);
            return NULL;
        }
        pl->parts[i] = p->id;
    }
    
    return pl;
}

static MxParticleList* particlelist_from_tuple(PyObject *tuple) {
    int nr_parts = PyTuple_Size(tuple);
    
    MxParticleList* pl = (MxParticleList*)PyType_GenericNew(&MxParticleList_Type, NULL, NULL);
    pl->flags = PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF;
    pl->size_parts = nr_parts;
    pl->parts = (int32_t*)malloc(nr_parts * sizeof(int32_t));;
    pl->nr_parts = nr_parts;
    
    for(int i = 0; i < nr_parts; ++i) {
        PyObject *obj = PyTuple_GET_ITEM(tuple, i);
        MxParticle *p = MxParticle_Get(obj);
        if(!p) {
            Py_DECREF(pl);
            return NULL;
        }
        pl->parts[i] = p->id;
    }
    
    return pl;
}


MxParticleList* MxParticleList_FromPyObject(PyObject *obj) {
    if(obj == NULL) {
        return NULL;
    }
    
    if(MxParticleList_Check(obj)) {
        Py_INCREF(obj);
        return (MxParticleList*)obj;
    }
    
    if(PyList_Check(obj) > 0) {
        return particlelist_from_list(obj);
    }
    
    if(PyTuple_Check(obj) > 0) {
        return particlelist_from_tuple(obj);
    }
    
    MxParticle *p = MxParticle_Get(obj);
    if(p) {
        MxParticleList* pl = (MxParticleList*)PyType_GenericNew(&MxParticleList_Type, NULL, NULL);
        pl->flags = PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF;
        pl->size_parts = 1;
        pl->parts = (int32_t*)malloc(pl->size_parts * sizeof(int32_t));;
        pl->nr_parts = pl->size_parts;
        pl->parts[0] = p->id;
        return pl;
    }
    
    return NULL;
}


// TODO: in universe.bind, check keywords are correct, and no extra keyworkds
// TODO: simulator init, turn off periodoc if only single cell.
PyObject* list_virial(MxParticleList *self, PyObject *args, PyObject *kwargs)
{
    try {
        
        Magnum::Matrix3 mat;
        
        HRESULT result = MxParticles_Virial(self->parts,
                                              self->nr_parts, 0, mat.data());
        
        if(SUCCEEDED(result)) {
            return mx::cast(mat);
        }
        else {
            // result if failed should set py error.
            return NULL;
        }
    }
    catch(const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

PyObject* list_radius_of_gyration(MxParticleList *self, PyObject *args, PyObject *kwargs) {
    try {
        float result;
        if(SUCCEEDED(MxParticles_RadiusOfGyration(self->parts, self->nr_parts, &result))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

PyObject* list_center_of_mass(MxParticleList *self, PyObject *args, PyObject *kwargs) {
    try {
        Magnum::Vector3 result;
        if(SUCCEEDED(MxParticles_CenterOfMass(self->parts, self->nr_parts, result.data()))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

PyObject* list_center_of_geometry(MxParticleList *self, PyObject *args, PyObject *kwargs) {
    try {
        Magnum::Vector3 result;
        if(SUCCEEDED(MxParticles_CenterOfGeometry(self->parts, self->nr_parts, result.data()))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

PyObject* list_moment_of_inertia(MxParticleList *self, PyObject *args, PyObject *kwargs) {
    try {
        Magnum::Matrix3 result;
        if(SUCCEEDED(MxParticles_MomentOfInertia(self->parts, self->nr_parts, result.data()))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

PyObject* list_positions(MxParticleList *self) {
    int nd = 2;
    
    int typenum = NPY_DOUBLE;
    
    npy_intp dims[] = {self->nr_parts,3};
    
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);
    
    double *data = (double*)PyArray_DATA(array);
    
    for(int i = 0; i < self->nr_parts; ++i) {
        MxParticle *part = _Engine.s.partlist[self->parts[i]];
        Magnum::Vector3 pos = part->global_position();
        data[i * 3 + 0] = pos.x();
        data[i * 3 + 1] = pos.y();
        data[i * 3 + 2] = pos.z();
    }
    
    return (PyObject*)array;
}

PyObject* list_spherical_positions(MxParticleList *self, PyObject *args, PyObject *kwargs) {

    Magnum::Vector3 origin;
    
    if(args && PyTuple_Size(args) > 0) {
        try {
            origin = mx::cast<Magnum::Vector3>(PyTuple_GetItem(args, 0));
        }
        catch(const std::exception &e) {
            c_error(E_FAIL, e.what());
            return NULL;
        }
    }
    else {
        Magnum::Vector3 center = {
            (float)_Engine.s.dim[0],
            (float)_Engine.s.dim[1],
            (float)_Engine.s.dim[2]
        };
        origin = center / 2;
    }
    
    int nd = 2;
    
    int typenum = NPY_DOUBLE;
    
    npy_intp dims[] = {self->nr_parts,3};
    
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);
    
    double *data = (double*)PyArray_DATA(array);
    
    for(int i = 0; i < self->nr_parts; ++i) {
        MxParticle *part = _Engine.s.partlist[self->parts[i]];
        Magnum::Vector3 pos = part->global_position();
        pos = MxCartesianToSpherical(pos, origin);
        data[i * 3 + 0] = pos.x();
        data[i * 3 + 1] = pos.y();
        data[i * 3 + 2] = pos.z();
    }
    
    return (PyObject*)array;
}

CAPI_FUNC(MxParticleList*) MxParticleList_Copy(const PyObject *obj) {
    MxParticleList *self = (MxParticleList*)obj;
    
    return MxParticleList_NewFromData(self->nr_parts, self->parts);
}

PyObject* list_copy(MxParticleList *self, PyObject *args, PyObject *kwargs) {
    return MxParticleList_Copy(self);
}


PyObject *MxParticleList_Pack(Py_ssize_t n, ...)
{
    Py_ssize_t i;
    MxParticleList *result;
    va_list vargs;
    
    va_start(vargs, n);
    result = MxParticleList_New(n);
    result->flags = PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF;
    result->nr_parts = n;
    if (result == NULL) {
        va_end(vargs);
        return NULL;
    }

    for (i = 0; i < n; i++) {
        int o = va_arg(vargs, int);
        result->parts[i] = o;
    }
    va_end(vargs);
    return result;
}

