/*
 * MxCluster.cpp
 *
 *  Created on: Aug 28, 2020
 *      Author: andy
 */

#include <MxCluster.hpp>

/* include some standard header files */
#include <stdlib.h>
#include <math.h>
#include <MxParticle.h>
#include "fptype.h"
#include <iostream>

// python type info
#include <structmember.h>
#include <MxNumpy.h>
#include <MxPy.h>
#include "engine.h"
#include "space.h"
#include "mx_runtime.h"
#include "space_cell.h"

#include <MxParticleEvent.h>
#include "../../rendering/NOMStyle.hpp"

#include <Magnum/Math/Distance.h>
#include <Magnum/Math/Matrix3.h>
#include <MxConvert.hpp>
#include <metrics.h>

struct MxParticleConstructor : PyObject {
    
};

PyObject *pctor_call(PyObject *, PyObject *, PyObject *) {
    return NULL;
}

/**
 * removes a particle from the list at the index.
 * returns null if not found
 */
static MxParticle *remove_particle_at_index(MxCluster *cluster, int index);

static PyObject* cluster_fission_plane(MxParticle *cluster, const Magnum::Vector4 &plane);

static PyObject* cluster_virial(PyObject *_self, PyObject *args, PyObject *kwargs);

static PyObject* cluster_radius_of_gyration(PyObject *_self, PyObject *args, PyObject *kwargs);

static PyObject* cluster_center_of_mass(PyObject *_self, PyObject *args, PyObject *kwargs);

static PyObject* cluster_center_of_geometry(PyObject *_self, PyObject *args, PyObject *kwargs);

static PyObject* cluster_moment_of_inertia(PyObject *_self, PyObject *args, PyObject *kwargs);




PyTypeObject MxParticleConstructor_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "ParticleConstructor",
    .tp_basicsize = sizeof(MxParticleConstructor),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           0,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           pctor_call,
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

static HRESULT pctor_type_init(PyObject *m) {
    if(PyType_Ready(&MxParticleConstructor_Type) < 0) {
        return mx_error(E_FAIL, "particle constructor PyType_Ready failed");
    }
    return S_OK;
}

typedef struct {
    lenfunc sq_length;
    binaryfunc sq_concat;
    ssizeargfunc sq_repeat;
    ssizeargfunc sq_item;
    void *was_sq_slice;
    ssizeobjargproc sq_ass_item;
    void *was_sq_ass_slice;
    objobjproc sq_contains;
    
    binaryfunc sq_inplace_concat;
    ssizeargfunc sq_inplace_repeat;
} xxx;

// sq_length
static Py_ssize_t cluster_length(PyObject *self) {
    std::cout << MX_FUNCTION << std::endl;
    MxParticle *part = MxParticle_Get(self);
    return part->nr_parts;
}

// sq_concat
static PyObject *cluster_concat(PyObject *, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_repeat
static PyObject *cluster_repeat(PyObject *, Py_ssize_t) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_item
static PyObject *cluster_item(PyObject *self, Py_ssize_t i) {
    std::cout << MX_FUNCTION << std::endl;
    MxParticle *part = MxParticle_Get(self);
    
    if (part) {
        if(i < part->nr_parts) {
            return _Engine.s.partlist[part->parts[i]]->py_particle();
        }
        else {
            PyErr_SetString(PyExc_IndexError, "cluster index out of range");
        }
    }
    return NULL;
}

// sq_ass_item
static int cluster_ass_item(PyObject *, Py_ssize_t, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_contains
static int cluster_contains(PyObject *, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_inplace_concat
static PyObject *cluster_inplace_concat(PyObject *, PyObject *) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

// sq_inplace_repeat
static PyObject *cluster_inplace_repeat(PyObject *, Py_ssize_t) {
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

PySequenceMethods MxCluster_Sequence = {
    .sq_length = (lenfunc)cluster_length,
    .sq_concat = cluster_concat,
    .sq_repeat = cluster_repeat,
    .sq_item = cluster_item,
    .was_sq_slice = NULL,
    .sq_ass_item = cluster_ass_item,
    .was_sq_ass_slice = NULL,
    .sq_contains = cluster_contains,
    .sq_inplace_concat = cluster_inplace_concat,
    .sq_inplace_repeat = cluster_inplace_repeat
};

MxParticleType *MxCluster_TypePtr;


//typedef PyObject *(*PyCFunctionWithKeywords)(PyObject *, PyObject *,
//PyObject *);
PyObject *cluster_particle_ctor(PyObject *a, PyObject *b, PyObject *c) {
    
    std::cout << "a: " << carbon::str(a) << std::endl;
    std::cout << "b: " << carbon::str(b) << std::endl;
    Py_RETURN_NONE;
}

PyMethodDef wrap = {
    .ml_name = "foo",   /* The name of the built-in function/method */
    .ml_meth = (PyCFunction)cluster_particle_ctor,    /* The C function that implements it */
    .ml_flags = METH_VARARGS | METH_KEYWORDS,   /* Combination of METH_xxx flags, which mostly
                                                 describe the args expected by the C func */
    .ml_doc = "docs"    /* The __doc__ attribute, or NULL */
};

static int cluster_init(MxParticleHandle *self, PyObject *_args, PyObject *_kwds) {
    std::cout << MX_FUNCTION << std::endl;
    
    int result = 0;
    
    MxParticleType *type = (MxParticleType*)self->ob_type;
    PyTypeObject *pytype = (PyTypeObject*)self->ob_type;
    
    PyTypeObject *base = pytype->tp_base;
    
    std::cout << "me: " << pytype->tp_name << std::endl;
    std::cout << "base: " << base->tp_name << std::endl;
    
    
    
    // call base class init
    if((result = ((PyTypeObject*)MxParticle_GetType())->tp_init(self, _args, _kwds)) != 0) {
        return result;
    }
    
    MxParticle *part = _Engine.s.partlist[self->id];
    
    part->flags |= PARTICLE_CLUSTER;
    
    //PyObject *
    //PyCFunction_NewEx(PyMethodDef *ml, PyObject *self, PyObject *module)
    
    PyObject *test = PyLong_FromLong(0);
    PyObject *func = PyCFunction_NewEx(&wrap, test, NULL);
    
    
    
    return result;
}

HRESULT MxClusterType_Init(MxParticleType *self, PyObject *_dict) {
    
    // itterate over all the items in the dict, and replace any particle
    // derived types with wrapper constructors.
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    
    PyObject *dict = ((PyTypeObject*)self)->tp_dict;
    
    while (PyDict_Next(dict, &pos, &key, &value)) {
        
        std::cout << "checking (" << carbon::str(key)
        << ", " << carbon::str(value) << ")" << std::endl;
        
        if(PyType_Check(value) && PyObject_IsSubclass(value, (PyObject*)MxParticle_GetType())) {
            std::cout << "found a particle type: " << carbon::str(key) << std::endl;
            
            PyObject *descr = MxClusterParticleCtor_New((MxParticleType*)self, (MxParticleType*)value);
            
            // if key is already in dict, PyDict_SetItem does not increase ref count
            if (PyDict_SetItem(dict, key, descr) < 0) {
                Py_DECREF(descr);
                return mx_error(E_FAIL, "failed to set cluster dictionary value for particle type");
            }
            Py_DECREF(descr);
            
            PyObject *o = PyDict_GetItem(dict, key);
            
            std::cout << "new obj " << carbon::str(o) << std::endl;
        }
    }
    return S_OK;
}

PyObject* cluster_fission_plane(MxParticle *cluster, const Magnum::Vector4 &plane) {
    
    Magnum::Debug() << MX_FUNCTION << ", plane: " << plane;
    
    // particles to move to daughter cluster.
    // only perform a split if the contained particles can be split into
    // two non-empty sets.
    std::vector<int> dparts;
    
    for(int i = 0; i < cluster->nr_parts; ++i) {
        MxParticle *p = cluster->particle(i);
        float dist = Magnum::Math::Distance::pointPlaneScaled(p->global_position(), plane);
        
        //Magnum::Debug() << "particle[" << i << "] position: " << p->global_position() << ", dist: " << dist;
        
        if(dist < 0) {
            dparts.push_back(p->id);
        }
    }
    
    if(dparts.size() > 0 && dparts.size() < cluster->nr_parts) {
        
        PyObject *_daughter = MxParticle_New((PyObject*)cluster->_pyparticle->ob_type,  NULL,  NULL);
        MxCluster *daughter = (MxCluster*)MxParticle_Get(_daughter);
        assert(daughter);
        
        std::cout << "split cluster "
        << cluster->id << " into ("
        << cluster->id << ":" << (cluster->nr_parts - dparts.size())
        << ", "
        << daughter->id << ": " << dparts.size() << ")" << std::endl;
        
        for(int i = 0; i < dparts.size(); ++i) {
            cluster->removepart(dparts[i]);
            daughter->addpart(dparts[i]);
        }
        
        return _daughter;
    }
    else {
        Py_RETURN_NONE;
    }
}

static PyObject* cluster_fission_normal_point(MxParticle *cluster,
    const Magnum::Vector3 &normal, const Magnum::Vector3 &point) {
    
    Magnum::Debug() << MX_FUNCTION << "normal: " << normal
                    << ", point: " << point << ", cluster center: "
                    << cluster->global_position();
    
    Magnum::Vector4 plane = Magnum::Math::planeEquation(normal, point);
    
    return cluster_fission_plane(cluster, plane);
}


static PyObject* cluster_fission_axis(MxParticle *cluster,
    const Magnum::Vector3 &axis) {
    
    Magnum::Debug() << MX_FUNCTION << "axis: " << axis;
    
    Magnum::Vector3 p1 = cluster->global_position();
    
    Magnum::Vector3 p2 = p1 + axis;
    
    Magnum::Vector3 p3 = p1 + MxRandomUnitVector();
    
    Magnum::Vector4 plane = Magnum::Math::planeEquation(p1, p2, p3);
    
    return cluster_fission_plane(cluster, plane);
}

int MxCluster_ComputeAggregateQuantities(struct MxCluster *cluster) {
    
    if(cluster->nr_parts <= 0) {
        return 0;
    }
    
    Magnum::Vector3 pos;
    
    // compute in global coordinates, particles can belong to different
    // space cells.
    /* Copy the position to x. */
    //for ( k = 0 ; k < 3 ; k++ )
    //    x[k] = s->partlist[id]->x[k] + s->celllist[id]
    
    for(int i = 0; i < cluster->nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[cluster->parts[i]];
        pos += p->global_position();
    }
    
    cluster->set_global_position(pos / cluster->nr_parts);
    
    return 0;
}


static PyObject* cluster_fission_random(MxParticle *cluster)
{
    PyObject *_daughter = MxParticle_New((PyObject*)cluster->_pyparticle->ob_type,  NULL,  NULL);
    
    MxCluster *daughter = (MxCluster*)MxParticle_Get(_daughter);
    assert(daughter);
    
    int halfIndex = cluster->nr_parts / 2;
    
    for(int i = halfIndex; i < cluster->nr_parts; ++i) {
        // adds to new cluster, sets id of contained particle.
        daughter->addpart(cluster->parts[i]);
        cluster->parts[i] = -1;
    }
    
    cluster->nr_parts = halfIndex;
    
    return _daughter;
}



/**
 # split the cell with a cleavage plane, in normal/point form.
 split(normal=[x, y, z], point=[px, py, pz])
 
 # split the cell with a cleavage plane normal, but use the clusters center as the point
 split(normal=[x, y, z])
 
 # if no named arguments are given, split interprets the first argument as a cleavage normal:
 split([x, y, z])
 
 # split using a cleavage *axis*, here, the split will generate a cleavage plane
 # that contains the given axis. This method is designed for the epiboly project,
 # where you’d give it an axis that’s the vector between the yolk center, and the
 # center of the cell. This will split the cell perpendicular to the yolk
 split(axis=[x, y, z])
 
 # default version of split uses a random cleavage plane that intersects the
 # cell center
 split()
 
 # the old style, were randomly picks contained objects, and assigns half of them
 # to the daughter cell
 split(random=True)
*/
static PyObject* cluster_fission(PyObject *_self, PyObject *args,
                                 PyObject *kwargs)
{
    std::cout << MX_FUNCTION << std::endl;
    
    MxParticle *cluster = MxParticle_Get(_self);
    
    if(!cluster) {
        PyErr_Format(PyExc_ValueError, "ERROR, given object is not a cluster");
        return NULL;
    }
    
    MxCluster_ComputeAggregateQuantities((MxCluster*)cluster);
    
    if(kwargs && PyDict_GetItemString(kwargs, "axis")) {
        // use axis form of split
        Magnum::Vector3 axis = mx::arg<Magnum::Vector3>("axis", 0, args, kwargs);
        return cluster_fission_axis(cluster, axis);
    }
    
    PyObject *a = NULL;
    if(kwargs && (a = PyDict_GetItemString(kwargs, "random")) && a == Py_True) {
        // use random form of split
        return cluster_fission_random(cluster);
    }
    
    Magnum::Vector3 normal;
    Magnum::Vector3 point;
    
    // check if being called as an event, with the first arge a time number
    a = NULL;
    if(args &&
       PyTuple_Check(args) &&
       PyTuple_Size(args) > 0 &&
       (a = PyTuple_GetItem(args, 0)) &&
       PyNumber_Check(a)) {
        float t = PyFloat_AsDouble(a);
        std::cout << "cluster split event(cluster id: " << cluster->id
                  << ", count: " << cluster->nr_parts
                  << ", time: " << t << ")" << std::endl;
        point = cluster->global_position();
        normal = MxRandomUnitVector();
    }
    else {
        // normal documented usage, grab args from args and kewords.
        normal = mx::arg("normal", 0, args, kwargs, MxRandomUnitVector());
        point = mx::arg("point", 1, args, kwargs, Magnum::Vector3{-1, -1, -1});
        
        std::cout << "using cleavage plane to split cluster" << std::endl;
        
        if(point[0] < 0 || point[1] < 0 || point[3] < 0) {
            point = cluster->global_position();
        }
    }
    
    return cluster_fission_normal_point(cluster, normal, point);
}

static PyMethodDef cluster_methods[] = {
    { "fission", (PyCFunction)cluster_fission, METH_VARARGS | METH_KEYWORDS, NULL },
    { "split", (PyCFunction)cluster_fission, METH_VARARGS | METH_KEYWORDS, NULL }, // alias name
    { "virial", (PyCFunction)cluster_virial, METH_VARARGS | METH_KEYWORDS, NULL },
    { "radius_of_gyration", (PyCFunction)cluster_radius_of_gyration, METH_VARARGS | METH_KEYWORDS, NULL },
    { "center_of_mass", (PyCFunction)cluster_center_of_mass, METH_VARARGS | METH_KEYWORDS, NULL },
    { "center_of_geometry", (PyCFunction)cluster_center_of_geometry, METH_VARARGS | METH_KEYWORDS, NULL },
    { "centroid", (PyCFunction)cluster_center_of_geometry, METH_VARARGS | METH_KEYWORDS, NULL },
    { "moment_of_inertia", (PyCFunction)cluster_moment_of_inertia, METH_VARARGS | METH_KEYWORDS, NULL },
    { "inertia", (PyCFunction)cluster_moment_of_inertia, METH_VARARGS | METH_KEYWORDS, NULL },
    { NULL, NULL, 0, NULL }
};



HRESULT cluster_type_init(PyObject *m)
{

    //make an instance of the base particle type, all new instances of base
    //class mechanica.Particle will be of this type
    PyTypeObject *ob = (PyTypeObject*)&engine::types[1];
    MxCluster_TypePtr = &engine::types[1];

    Py_TYPE(ob) =          &MxParticleType_Type;
    ob->tp_base =          (PyTypeObject*)&engine::types[0];
    ob->tp_getset =        0;
    ob->tp_methods =       cluster_methods;
    ob->tp_name =          "Cluster";
    ob->tp_basicsize =     sizeof(MxParticleHandle);
    ob->tp_flags =         Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    ob->tp_doc =           "Custom objects";
    ob->tp_init =          (initproc)cluster_init;
    ob->tp_new =           0;
    ob->tp_del =           [] (PyObject *p) -> void {
        std::cout << "tp_del MxCluster" << std::endl;
    };
    ob->tp_finalize =      [] (PyObject *p) -> void {
        std::cout << "tp_finalize MxCluster" << std::endl;
    };



    if(PyType_Ready(ob) < 0) {
        return mx_error(E_FAIL, "PyType_Ready on MxCluster failed");
    }

    MxParticleType *pt = (MxParticleType*)ob;
    
    pt->parts.init();
    pt->radius = 1.0;
    pt->minimum_radius = 0.0;
    pt->mass = 1.0;
    pt->charge = 0.0;
    pt->id = 0;
    pt->dynamics = PARTICLE_NEWTONIAN;

    // TODO: default particle style...
    pt->style = NOMStyle_New(NULL, NULL);
    pt->style->color = Magnum::Color3::fromSrgb(MxParticle_Colors[1]);

    ::strncpy(pt->name, "Cluster", MxParticleType::MAX_NAME);
    ::strncpy(pt->name2, "Cluster", MxParticleType::MAX_NAME);

    // set the singlton particle type data to the new item here.
    if (PyModule_AddObject(m, "Cluster", (PyObject*)ob) < 0) {
        return E_FAIL;
    }

    std::cout << "added Cluster to mechanica module" << std::endl;

    engine::nr_types = 2;

    return S_OK;
}

// TODO: merge this with static PyObject* random_point_solidsphere(int n)
static Magnum::Vector3 random_point_solid_sphere(float radius) {

    std::uniform_real_distribution<double> uniform01(0.0, 1);
    

        double theta = 2 * M_PI * uniform01(CRandom);
        double phi = acos(1 - 2 * uniform01(CRandom));
        double r = std::cbrt(uniform01(CRandom)) * radius;
        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);
    
    return Magnum::Vector3{x, y, z};
}



PyObject *pctor_wrapper_func(PyObject *self, PyObject *args,
                void *wrapped, PyObject *kwds) {
    std::cout << "self: " << carbon::str(self) << std::endl;
    std::cout << "args: " << carbon::str(args) << std::endl;
    std::cout << "kwds: " << carbon::str(kwds) << std::endl;
    std::cout << "wrapped: " << carbon::str((PyObject*)wrapped) << std::endl;
    
    if(kwds) {
        Py_INCREF(kwds);
    }
    else {
        kwds = PyDict_New();
    }
    
    PyDict_SetItemString(kwds, "cluster", self);
    
    // don't use particle pointer, adding new child particles
    // can move it.
    MxParticle *p = MxParticle_Get(self);
    assert(p);
    int clusterId = p->id;
    float radius = p->radius;
    p = NULL;
    PyObject *result = NULL;

    // type of nested particle
    PyTypeObject *ptype = (PyTypeObject*)wrapped;
    
    // first item is a number, use this as a count
    if(PyTuple_Size(args) > 0 && PyLong_Check(PyTuple_GetItem(args, 0))) {
        int count = PyLong_AsLong(PyTuple_GetItem(args, 0));
        PyObject *newArgs = PyTuple_GetSlice(args, 1, PyTuple_Size(args));
        
        for(int i = 0; i < count; ++i) {
            // adds a particle to our particles list
            
            // set postion
            Magnum::Vector3 pos;
            space_getpos(&_Engine.s, clusterId, pos.data());
            pos = random_point_solid_sphere(radius) + pos;
            PyObject *pypos = mx::cast(pos);
            PyDict_SetItemString(kwds, "position", pypos);
            PyObject *part = PyObject_Call((PyObject*)ptype, newArgs, kwds);
            assert(part);
            Py_DECREF(part);
            Py_DECREF(pypos);
        }
        
        Py_DecRef(newArgs);
        result = Py_None;
        Py_INCREF(result);
    }
    else {
        result = PyObject_Call((PyObject*)ptype, args, kwds);
    }
    
    Py_DECREF(kwds);
    return result;
}

wrapperbase pctor_wrapper = {
    .name = "ClusterParticleConstructor",
    .offset = 0,
    .function = 0,
    .wrapper = (wrapperfunc)pctor_wrapper_func,
    .doc = "foo",
    .flags = PyWrapperFlag_KEYWORDS,
    .name_strobj = 0
};

MxParticleType *MxParticleType_Get(PyObject *obj) {
    if(obj == NULL) {
        return NULL;
    }
    if(PyType_Check(obj)) {
        int res = PyObject_IsSubclass(obj, (PyObject*)MxParticle_GetType());
        if (res < 0) {
            PyErr_Print();
        }
        if(res > 0) {
            return (MxParticleType*)obj;
        }
    }
    if(PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType())) {
        return (MxParticleType*)obj->ob_type;
    }
    else if(PyObject_IsInstance(obj, (PyObject*)&PyWrapperDescr_Type)) {
        PyWrapperDescrObject *descr = (PyWrapperDescrObject*)obj;
        if(descr->d_base == &pctor_wrapper) {
            return (MxParticleType*)descr->d_wrapped;
        }
    }
    return NULL;
}

PyObject *makeThing() {
    PyObject *test = PyLong_FromLong(1234567890);
    
    //return PyDescr_NewMethod((PyTypeObject*)&engine::types[1], &wrap);
    //PyObject *func = PyCFunction_NewEx(&wrap, NULL, NULL);
    
    //return func;
    
    return PyDescr_NewWrapper((PyTypeObject*)&engine::types[1], &pctor_wrapper, test);
    
}

PyObject *MxClusterParticleCtor_New(
    MxParticleType *clusterType, MxParticleType *containedParticleType) {
    return PyDescr_NewWrapper((PyTypeObject*)clusterType, &pctor_wrapper, containedParticleType);
}


/**
 * adds an existing particle to the cluster.
 */
int MxCluster_AddParticle(struct MxCluster *cluster, struct MxParticle *part) {
    return -1;
}

/**
 * creates a new particle, and adds it to the cluster.
 */
PyObject* MxCluster_CreateParticle(PyObject *self,
                                   PyObject* particleType, PyObject *args, PyObject *kwargs) {
    return NULL;
}


MxParticle *remove_particle_at_index(MxCluster *cluster, int index) {
    if(index >= cluster->nr_parts) {
        return NULL;
    }
    
    int pid = cluster->parts[index];
    
    for(int i = index; i + 1 < cluster->nr_parts; ++i) {
        cluster->parts[i] = cluster->parts[i+i];
    }
    
    cluster->nr_parts -= 1;
    cluster->parts[cluster->nr_parts] = -1;
    
    MxParticle *part = _Engine.s.partlist[pid];
    
    part->clusterId = -1;
    
    return part;
}

HRESULT _MxCluster_init(PyObject *m) {
    std::cout << MX_FUNCTION << std::endl;
    return cluster_type_init(m);
}

int MxCluster_Check(PyObject *p) {
    return p && PyObject_IsSubclass((PyObject*)p->ob_type, (PyObject*)MxCluster_GetType());
}


// TODO: in universe.bind, check keywords are correct, and no extra keyworkds
// TODO: simulator init, turn off periodoc if only single cell. 
PyObject* cluster_virial(PyObject *_self, PyObject *args, PyObject *kwargs)
{
    try {
        MxParticle *self = MxParticle_Get(_self);
        
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
        C_RETURN_EXP(e);
    }
}

PyObject* cluster_radius_of_gyration(PyObject *_self, PyObject *args, PyObject *kwargs) {
    try {
        MxParticle *self = MxParticle_Get(_self);
        float result;
        if(SUCCEEDED(MxParticles_RadiusOfGyration(self->parts, self->nr_parts, &result))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject* cluster_center_of_mass(PyObject *_self, PyObject *args, PyObject *kwargs) {
    try {
        MxParticle *self = MxParticle_Get(_self);
        Magnum::Vector3 result;
        if(SUCCEEDED(MxParticles_CenterOfMass(self->parts, self->nr_parts, result.data()))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject* cluster_center_of_geometry(PyObject *_self, PyObject *args, PyObject *kwargs) {
    try {
        MxParticle *self = MxParticle_Get(_self);
        Magnum::Vector3 result;
        if(SUCCEEDED(MxParticles_CenterOfGeometry(self->parts, self->nr_parts, result.data()))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject* cluster_moment_of_inertia(PyObject *_self, PyObject *args, PyObject *kwargs) {
    try {
        MxParticle *self = MxParticle_Get(_self);
        Magnum::Matrix3 result;
        if(SUCCEEDED(MxParticles_MomentOfInertia(self->parts, self->nr_parts, result.data()))) {
            return mx::cast(result);
        }
        return NULL;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}
