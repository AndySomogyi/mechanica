/*
 * MxCutPlane.cpp
 *
 *  Created on: Mar 26, 2021
 *      Author: andy
 */

#include "MxClipPlane.hpp"
#include <Magnum/Math/Distance.h>
#include <MxConvert.hpp>
#include <MxSimulator.h>
#include <rendering/MxUniverseRenderer.h>


struct MxClipPlane : PyObject
{
    int index;
    MxClipPlane(int i);
};

struct MxClipPlanes : PyObject {
    MxClipPlanes();
};

MxClipPlanes _clipPlanesObj;

PyTypeObject MxClipPlane_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ClipPlane",
    .tp_basicsize = sizeof(MxClipPlane),
    .tp_itemsize =       0,
    .tp_dealloc =        (destructor)0,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
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
    .tp_methods =        0,
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

MxClipPlane::MxClipPlane(int i) {
    index = i;
    ob_refcnt = 1;
    ob_type = &MxClipPlane_Type;
}

static Magnum::Vector4 parse_plane_obj(PyObject *o) {
    // treating it like a plane equation
    if(PySequence_Check(o) && PySequence_Size(o) == 4) {
        Magnum::Vector4 pe = mx::cast<Magnum::Vector4>(o);
        return pe;
    }
    else {
        if(!PyTuple_Check(o)) {
            std::string msg = "clip plane argumennts must be a list of tuples, but recieved: ";
            msg += carbon::str(o);
            throw std::logic_error(msg.c_str());
        }
        
        if(PyTuple_Size(o) != 2) {
            std::string msg = "each clip plane item must be a tuple of (point, normal), or length 4 vector, instead got: ";
            msg += carbon::str(o);
            throw std::logic_error(msg.c_str());
        }
        
        Magnum::Vector3 point = mx::cast<Magnum::Vector3>(PyTuple_GetItem(o, 0));
        Magnum::Vector3 normal = mx::cast<Magnum::Vector3>(PyTuple_GetItem(o, 1));
        Magnum::Vector4 pe = Magnum::Math::planeEquation(normal, point);
        return pe;
    }
}

// sq_length
static Py_ssize_t cp_length(PyObject *_self) {
    MxSimulator *sim = MxSimulator::Get();
    MxUniverseRenderer *renderer = sim->getRenderer();
    return renderer->clipPlaneCount();
}

// sq_item
static PyObject *cp_subscript(PyObject *_self, PyObject *pindex) {
    try {
        MxSimulator *sim = MxSimulator::Get();
        MxUniverseRenderer *renderer = sim->getRenderer();
        
        unsigned index = mx::cast<unsigned>(pindex);
        if(index > renderer->clipPlaneCount()) {
            throw std::range_error("index out of bounds");
        }
        return mx::cast(renderer->getClipPlaneEquation(index));
    }
    catch(const std::exception &e) {
        C_ERR(E_FAIL, e.what());
        return NULL;
    }
}

// sq_ass_item
static int cp_ass_item(PyObject *_self, PyObject *pindex, PyObject *o) {
    try {
        MxSimulator *sim = MxSimulator::Get();
        MxUniverseRenderer *renderer = sim->getRenderer();
        
        unsigned index = mx::cast<unsigned>(pindex);
        if(index > renderer->clipPlaneCount()) {
            throw std::range_error("index out of bounds");
        }
        
        Magnum::Vector4 pe = parse_plane_obj(o);
        renderer->setClipPlaneEquation(index, pe);
        
        return 0;
    }
    catch(const std::exception &e) {
        C_ERR(E_FAIL, e.what());
        return -1;
    }
}

static PyMappingMethods mapping = {
    cp_length,      //mp_length
    cp_subscript,   //mp_subscript
    cp_ass_item,    //mp_ass_subscript
};

static PyMethodDef methods[] = {
    { NULL, NULL, 0, NULL }
};

PyTypeObject MxClipPlanes_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ClipPlanes",
    .tp_basicsize = sizeof(MxClipPlanes),
    .tp_itemsize =       0,
    .tp_dealloc =        (destructor)0,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     &mapping,
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
    .tp_methods =        methods,
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


MxClipPlanes::MxClipPlanes() {
    ob_refcnt = 1;
    ob_type = &MxClipPlanes_Type;
}


/**
 * get a borrowed reference to the cut planes collection.
 */
PyObject *MxClipPlanes_Get() {
    Py_INCREF(&_clipPlanesObj);
    return &_clipPlanesObj;
}

std::vector<Magnum::Vector4> MxClipPlanes_ParseConfig(PyObject *clipPlanes) {
    try {
        std::vector<Magnum::Vector4> equations;
        
        carbon::sequence seq = carbon::cast<carbon::sequence>(clipPlanes);
        
        for(int i = 0; i < seq.size(); ++i) {
            PyObject *o = seq.get(i);
            
            equations.push_back(parse_plane_obj(o));
        }
        
        return equations;
    }
    catch(const std::exception &e) {
        throw std::logic_error(std::string("could not parse clip_planes argument: ") + e.what());
    }
}


/**
 * internal function to initalize the cut plane types
 */
HRESULT _MxClipPlane_Init(PyObject *m) {

    if (PyType_Ready((PyTypeObject*)&MxClipPlane_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&MxClipPlane_Type);

    if (PyType_Ready((PyTypeObject*)&MxClipPlanes_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&MxClipPlanes_Type);

    return S_OK;
}

