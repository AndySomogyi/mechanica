/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/


/* include some standard header files */
#include <stdlib.h>
#include <math.h>
#include <MxParticle.h>
#include "fptype.h"
#include <iostream>

// python type info
#include <structmember.h>
#include <MxNumpy.h>

#include <pybind11/pybind11.h>

#include <MxPy.h>
#include "engine.h"
#include "space.h"




struct Foo {
    int x; int y; int z;
};


//template <typename C, typename D, typename... Extra>
//class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {


    
    
template<typename C, typename T>
void f(T C::*pm)
{
    std::cout << "sizeof pm: " << sizeof(pm) << std::endl;
    std::cout << "sizeof T: " << sizeof(T) << std::endl;
    //std::cout << "sizeof *pm: " << sizeof(MxParticle::*pm) << std::endl;
    std::cout << typeid(T).name() << std::endl;
    
    if(std::is_same<T, float>::value) {
        std::cout << "is float" << std::endl;
    }
    
    if(std::is_same<T, Magnum::Vector3>::value) {
        std::cout << "is vec" << std::endl;
    }
    
    std::cout << "offset of: " << offset_of(pm);
}

/**
 * initialize a newly allocated type
 *
 * adds a new data entry to the engine.
 */
static HRESULT MxParticleType_Init(MxParticleType *self, PyObject *dict);


 



static PyObject *particle_getattro(PyObject* obj, PyObject *name) {
    
    PyObject *s = PyObject_Str(name);
    PyObject* pyStr = PyUnicode_AsEncodedString(s, "utf-8", "Error ~");
    const char *cstr = PyBytes_AS_STRING(pyStr);
    std::cout << obj->ob_type->tp_name << ": " << __PRETTY_FUNCTION__ << ":" << cstr << "\n";
    return PyObject_GenericGetAttr(obj, name);
}






struct Offset {
    uint32_t kind;
    uint32_t offset;
};

static_assert(sizeof(Offset) == sizeof(void*), "error, void* must be 64 bit");

static_assert(sizeof(MxGetSetDefInfo) == sizeof(void*), "error, void* must be 64 bit");
static_assert(sizeof(MxGetSetDef) == sizeof(PyGetSetDef), "error, void* must be 64 bit");

PyObject * vector4_getter(MxParticle *obj, void *closure) {
    void* pClosure = &closure;
    Offset o = *(Offset*)pClosure;

    char* pVec = ((char*)obj) + o.offset;

    Magnum::Vector4 *vec = (Magnum::Vector4 *)pVec;

    pybind11::handle h = pybind11::cast(vec).release();
    
    //pybind11::object h2 = pybind11::cast(*vec);
    

    
    PyObject *result = h.ptr();
    
    std::cout << "result: " << result << std::endl;
    std::cout << "result.refcnt: " << result->ob_refcnt << std::endl;
    std::cout << "result.type: " << result->ob_type->tp_name << std::endl;

    return result;

}

int vector4_setter(PyObject *, PyObject *, void *) {

    return -1;

}



::PyGetSetDef create_vector4_getset() {

    ::PyGetSetDef result;

    result.name = "foo";
    result.get = (getter)vector4_getter;
    result.set = vector4_setter;
    result.doc = "docs";

    Offset o = {0, offsetof(MxParticle, position)};

    void** p = (void**)&o;


    result.closure = (void*)(*p);


    return result;

}


PyGetSetDef gsd = {
        .name = "descr",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            const char* on = obj != NULL ? obj->ob_type->tp_name : "NULL";
            std::cout << "getter(obj.type:" << on << ", p:" << p << ")" << std::endl;

            bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
            bool isParticleType = PyObject_IsInstance(obj, (PyObject*)&MxParticleType_Type);

            std::cout << "is particle: " << isParticle << std::endl;
            std::cout << "is particle type: " << isParticleType << std::endl;
            return PyLong_FromLong(567);
        },
        .set = [](PyObject *obj, PyObject *, void *p) -> int {
            const char* on = obj != NULL ? obj->ob_type->tp_name : "NULL";
            std::cout << "setter(obj.type:" << on << ", p:" << p << ")" << std::endl;

            bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
            bool isParticleType = PyObject_IsInstance(obj, (PyObject*)&MxParticleType_Type);

            std::cout << "is particle: " << isParticle << std::endl;
            std::cout << "is particle type: " << isParticleType << std::endl;

            return 0;
        },
        .doc = "test doc",
        .closure = NULL
    };

PyGetSetDef gs_charge = {
    .name = "charge",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->data->charge).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        
        try {
            double *x = &type->data->charge;
            *x = pybind11::cast<double>(val);
            return 0;
        }
        catch (const pybind11::builtin_exception &e) {
            e.set_error();
            return -1;
        }
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_mass = {
    .name = "mass",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->data->mass).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        
        try {
            double *x = &type->data->mass;
            *x = pybind11::cast<double>(val);
            return 0;
        }
        catch (const pybind11::builtin_exception &e) {
            e.set_error();
            return -1;
        }
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_name = {
    .name = "name",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->data->name).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        PyErr_SetString(PyExc_PermissionError, "read only");
        return -1;
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_name2 = {
    .name = "name2",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)&MxParticle_Type);
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->data->name2).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        PyErr_SetString(PyExc_PermissionError, "read only");
        return -1;
    },
    .doc = "test doc",
    .closure = NULL
};



PyGetSetDef particle_getsets[] = {
    gs_charge,
    gs_mass,
    gs_name,
    gs_name2,
    gsd,
    {
        .name = "position",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            int id = ((MxPyParticle*)obj)->part->id;
            Magnum::Vector3 vec;
            space_getpos(&_Engine.s, id, vec.data());
            return pybind11::cast(vec).release().ptr();
            
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxPyParticle*)obj)->part->id;
                Magnum::Vector3 vec = pybind11::cast<Magnum::Vector3>(val);
                space_setpos(&_Engine.s, id, vec.data());
                return 0;
            }
            catch (const pybind11::builtin_exception &e) {
                e.set_error();
                return -1;
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "velocity",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            Magnum::Vector3 *vec = &((MxPyParticle*)obj)->part->velocity;
            return pybind11::cast(vec).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                Magnum::Vector3 *vec = &((MxPyParticle*)obj)->part->velocity;
                *vec = pybind11::cast<Magnum::Vector3>(val);
                return 0;
            }
            catch (const pybind11::builtin_exception &e) {
                e.set_error();
                return -1;
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "force",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            Magnum::Vector3 *vec = &((MxPyParticle*)obj)->part->force;
            return pybind11::cast(vec).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                Magnum::Vector3 *vec = &((MxPyParticle*)obj)->part->force;
                *vec = pybind11::cast<Magnum::Vector3>(val);
                return 0;
            }
            catch (const pybind11::builtin_exception &e) {
                e.set_error();
                return -1;
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "id",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            int x = ((MxPyParticle*)obj)->part->id;
            return pybind11::cast(x).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int *x = &((MxPyParticle*)obj)->part->id;
                *x = pybind11::cast<int>(val);
                return 0;
            }
            catch (const pybind11::builtin_exception &e) {
                e.set_error();
                return -1;
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "type_id",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            int x = ((MxPyParticle*)obj)->part->typeId;
            return pybind11::cast(x).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                short *x = &((MxPyParticle*)obj)->part->typeId;
                *x = pybind11::cast<short>(val);
                return 0;
            }
            catch (const pybind11::builtin_exception &e) {
                e.set_error();
                return -1;
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "flags",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            unsigned short x = ((MxPyParticle*)obj)->part->flags;
            return pybind11::cast(x).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                unsigned short *x = &((MxPyParticle*)obj)->part->flags;
                *x = pybind11::cast<unsigned short>(val);
                return 0;
            }
            catch (const pybind11::builtin_exception &e) {
                e.set_error();
                return -1;
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

static PyObject* particle_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    //std::cout << MX_FUNCTION << ", type: " << type->tp_name << std::endl;
    return PyType_GenericNew(type, args, kwargs);
}

static int particle_init(MxPyParticle *self, PyObject *_args, PyObject *_kwds) {
    // std::cout << MX_FUNCTION << std::endl;
    
    MxParticleType *type = (MxParticleType*)self->ob_type;
    
    MxParticle part = {
        .position = {},
        .velocity = {},
        .force = {},
        .typeId = type->data->id,
        .id = _Engine.s.nr_parts
    };
    
    
    try {
        pybind11::detail::loader_life_support ls{};
        pybind11::args args = pybind11::reinterpret_borrow<pybind11::args>(_args);
        pybind11::kwargs kwargs = pybind11::reinterpret_borrow<pybind11::kwargs>(_kwds);
        
        part.position = arg<Magnum::Vector3>("position", 0, args.ptr(), kwargs.ptr(), Magnum::Vector3{});
        part.velocity = arg<Magnum::Vector3>("velocity", 1, args.ptr(), kwargs.ptr(), Magnum::Vector3{});
        
        MxParticle *p = NULL;
        double pos[] = {part.position[0], part.position[1], part.position[2]};
        int result = space_addpart (&_Engine.s, &part, pos, &p);
        
        self->part = p;
        
        return 0;
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return -1;
    }
}


/**
 * The base particle type
 * this instance points to the 0'th item in the global engine struct.
 */
MxParticleType MxParticle_Type = {
{
  {
      PyVarObject_HEAD_INIT(NULL, 0)
      .tp_name =           "Particle",
      .tp_basicsize =      sizeof(MxPyParticle),
      .tp_itemsize =       0, 
      .tp_dealloc =        0, 
      .tp_print =          0, 
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
      .tp_flags =          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
      .tp_doc =            "Custom objects",
      .tp_traverse =       0, 
      .tp_clear =          0, 
      .tp_richcompare =    0, 
      .tp_weaklistoffset = 0, 
      .tp_iter =           0, 
      .tp_iternext =       0, 
      .tp_methods =        0, 
      .tp_members =        0,
      .tp_getset =         particle_getsets,
      .tp_base =           0, 
      .tp_dict =           0, 
      .tp_descr_get =      0, 
      .tp_descr_set =      0, 
      .tp_dictoffset =     0, 
      .tp_init =           (initproc)particle_init,
      .tp_alloc =          0, 
      .tp_new =            particle_new,
      .tp_free =           0, 
      .tp_is_gc =          0, 
      .tp_bases =          0, 
      .tp_mro =            0, 
      .tp_cache =          0, 
      .tp_subclasses =     0, 
      .tp_weaklist =       0, 
      .tp_del =            [] (PyObject *p) -> void {
          std::cout << "tp_del MxPyParticle" << std::endl;
      },
      .tp_version_tag =    0, 
      .tp_finalize =       [] (PyObject *p) -> void {
          // std::cout << "tp_finalize MxPyParticle" << std::endl;
      }
    }
  },
    .data = NULL // when the engine is intialized, it sets this pointer to the
                 // first element in the types list.
};



static getattrofunc savedFunc = NULL;

static PyObject *particle_type_getattro(PyObject* obj, PyObject *name) {
    
    PyObject *s = PyObject_Str(name);
    PyObject* pyStr = PyUnicode_AsEncodedString(s, "utf-8", "Error ~");
    const char *cstr = PyBytes_AS_STRING(pyStr);
    //std::cout << obj->ob_type->tp_name << ": " << __PRETTY_FUNCTION__ << ":" << cstr << "\n";
    return savedFunc(obj, name);
}





static PyObject *
particle_type_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    std::string t = pybind11::cast<pybind11::str>((PyObject*)type);
    std::string a = pybind11::cast<pybind11::str>(args);
    std::string k = pybind11::cast<pybind11::str>(kwds);
    
    std::cout << MX_FUNCTION << "(type: " << t << ", args: " << a << ", kwargs: " << k << ")" << std::endl;
    
    PyTypeObject *result;
    PyObject *fields;

    /* create the new instance (which is a class,
           since we are a metatype!) */
    result = (PyTypeObject *)PyType_Type.tp_new(type, args, kwds);

    if (!result)
        return NULL;

    return (PyObject*)result;
}

/*
 *   ID of this type
    int id;

    /** Constant physical characteristics
    double mass, imass, charge;

    /** Nonbonded interaction parameters.
    double eps, rmin;

    /** Name of this paritcle type.
    char name[64], name2[64];
 */


static PyGetSetDef particle_type_getset[] = {
    gsd,
    gs_charge,
    gs_mass,
    gs_name,
    gs_name2,
    {NULL},
};

static PyObject *
particle_type_descr_get(PyMemberDescrObject *descr, PyObject *obj, PyObject *type)
{
    return PyType_Type.tp_descr_get((PyObject*)descr, obj, type);
}

static int particle_type_init(MxParticleType *self, PyObject *_args, PyObject *kwds) {
    
    std::string s = pybind11::cast<pybind11::str>((PyObject*)self);
    std::string a = pybind11::cast<pybind11::str>(_args);
    std::string k = pybind11::cast<pybind11::str>(kwds);
    
    std::cout << MX_FUNCTION << "(self: " << s << ", args: " << a << ", kwargs: " << k << ")" << std::endl;
    
    //args is a tuple of (name, (bases, .), dict),
    pybind11::tuple args = pybind11::reinterpret_borrow<pybind11::tuple>(_args);
    
    pybind11::str name = args[0];
    pybind11::tuple bases = args[1];
    pybind11::object dict = args[2];

    return MxParticleType_Init(self, dict.ptr());
}

/**
 * particle type metatype
 */
PyTypeObject MxParticleType_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "ParticleType",
    .tp_basicsize =      sizeof(MxParticleType),
    .tp_itemsize =       0, 
    .tp_dealloc =        0, 
    .tp_print =          0, 
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
    .tp_flags =          Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc =            "Custom objects",
    .tp_traverse =       0, 
    .tp_clear =          0, 
    .tp_richcompare =    0, 
    .tp_weaklistoffset = 0, 
    .tp_iter =           0, 
    .tp_iternext =       0, 
    .tp_methods =        0, 
    .tp_members =        0,
    .tp_getset =         particle_type_getset,
    .tp_base =           0, 
    .tp_dict =           0, 
    .tp_descr_get =      (descrgetfunc)particle_type_descr_get,
    .tp_descr_set =      0, 
    .tp_dictoffset =     0, 
    .tp_init =           (initproc)particle_type_init,
    .tp_alloc =          0, 
    .tp_new =            particle_type_new,
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




/** ID of the last error */
int particle_err = PARTICLE_ERR_OK;



static void printTypeInfo(const char* name, PyTypeObject *p) {
    
    printf("%s : {"
           "typename: %s, \n"
           "baseTypeName: %s \n }\n",
           name,
           Py_TYPE(p)->tp_name,
           p->tp_base->tp_name);
    
    /*
    if(p->tp_getattro) {
        PyObject *o = PyUnicode_FromString("foo");
        p->tp_getattro((PyObject*)p, o);
    }
     */
}

HRESULT _MxParticle_init(PyObject *m)
{
    
    
    
    f<>(&MxParticle::q);
    
    f<>(&MxParticle::position);
    
    f<>(&MxParticle::x);
    


    /*************************************************
     *
     * Metaclasses first
     */

    //PyCStructType_Type.tp_base = &PyType_Type;
    // if (PyType_Ready(&PyCStructType_Type) < 0)
    //     return NULL;
    MxParticleType_Type.tp_base = &PyType_Type;
    if (PyType_Ready((PyTypeObject*)&MxParticleType_Type) < 0) {
        std::cout << "could not initialize MxParticleType_Type " << std::endl;
        return E_FAIL;
    }
    
    printTypeInfo("MxParticleType_Type", &MxParticleType_Type);
    

    

    /*************************************************
     *
     * Classes using a custom metaclass second
     */
    // Py_TYPE(&Struct_Type) = &PyCStructType_Type;
    // Struct_Type.tp_base = &PyCData_Type;
    // if (PyType_Ready(&Struct_Type) < 0)
    //     return NULL;
    Py_TYPE(&MxParticle_Type) = &MxParticleType_Type;
    //MxParticle_Type.tp_base = &PyBaseObject_Type;
    if (PyType_Ready((PyTypeObject*)&MxParticle_Type) < 0) {
        std::cout << "could not initialize MxParticle_Type " << std::endl;
        return E_FAIL;
    }
    



    //MxParticleType_Type.tp_dict

    //PyDict_SetItemString(MxParticleType_Type.tp_dict, "descr", descr);

    //descr = PyDescr_NewGetSet((PyTypeObject*)&MxParticleType_Type, &gsd);

    //PyDict_SetItemString(MxParticle_Type.ht_type.tp_dict, "descr", descr);





    printTypeInfo("MxParticle_Type", (PyTypeObject*)&MxParticle_Type);
    
    
    if(MxParticleType_Type.tp_getattro) {
        savedFunc = MxParticleType_Type.tp_getattro;
        MxParticleType_Type.tp_getattro = particle_type_getattro;
    }
    

    Py_INCREF(&MxParticleType_Type);
    if (PyModule_AddObject(m, "ParticleType", (PyObject *)&MxParticleType_Type) < 0) {
        Py_DECREF(&MxParticleType_Type);
        return E_FAIL;
    }

    Py_INCREF(&MxParticle_Type);
    if (PyModule_AddObject(m, "Particle", (PyObject *)&MxParticle_Type) < 0) {
        Py_DECREF(&MxParticle_Type);
        return E_FAIL;
    }

    return S_OK;
}

int MxParticleCheck(PyObject *o)
{
    return -1;
}

MxPyParticle* MxPyParticle_New(MxParticle *data)
{
    PyTypeObject *type = (PyTypeObject*)_Engine.types[data->typeId].pyType;
    MxPyParticle *part = (MxPyParticle*)PyType_GenericAlloc(type, 0);
    part->part = data;
    return part;
}


MxParticleType* MxParticleType_New(const char *_name, PyObject *dict)
{
    // static PyObject *
    // type_call(PyTypeObject *type, PyObject *args, PyObject *kwds)
    
    PyTypeObject *p = (PyTypeObject*)&MxParticle_Type;
    
    pybind11::str name(_name);
    pybind11::tuple bases(1);
    bases[0] = (PyObject*)&MxParticle_Type;
    pybind11::tuple args(3);
    args[0] = name;
    args[1] = bases;
    args[2] = dict;

    MxParticleType *result = (MxParticleType*)PyType_Type.tp_call((PyObject*)&PyType_Type, args.ptr(), NULL);

    assert(result && PyType_IsSubtype((PyTypeObject*)result, (PyTypeObject*)&MxParticle_Type));

    return result;
}

HRESULT MxParticleType_Init(MxParticleType *self, PyObject *_dict)
{
    double mass = 1.0;
    double charge = 0.0;
    std::string name2;
    
    try {
        pybind11::dict dict = pybind11::reinterpret_borrow<pybind11::dict>(_dict);
        if(dict.contains("mass")) {
            mass = dict["mass"].cast<double>();
        }

        if(dict.contains("charge")) {
            charge = dict["charge"].cast<double>();
        }
        
        if(dict.contains("name2")) {
            name2 = dict["name2"].cast<std::string>();
        }

        int er = engine_addtype_for_type(&_Engine, mass,
                charge, self->ht_type.tp_name , name2.c_str(), self);
        
        // pybind does not seem to wrap deleting item from dict, WTF?!?
        if(self->ht_type.tp_dict) {
            
            PyObject *_dict = self->ht_type.tp_dict;
            
            pybind11::object key = pybind11::cast("mass");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("charge");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("name2");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
        }
    

        return er >= 0 ? S_OK : c_error(CERR_FAIL, "failed to add new type to engine");
    }
    catch(const std::exception &e) {
        return c_error(CERR_EXCEP, e.what());
    }

    return CERR_FAIL;
}

MxParticleType* MxParticleType_ForEngine(struct engine *e, double mass,
        double charge, const char *name, const char *name2)
{
    pybind11::dict dict;

    dict["mass"] = pybind11::cast(mass);
    dict["charge"] = pybind11::cast(charge);
    
    if(name2) {
        dict["name2"] = pybind11::cast(name2);
    }

    return MxParticleType_New(name, dict.ptr());
}

