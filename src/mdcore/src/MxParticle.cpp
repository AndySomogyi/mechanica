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

PyObject *test() {

    pybind11::module m;


    pybind11::class_<Foo>(m, "Pet")
        .def_readwrite("name", &Foo::x);

    int i = 0;

    pybind11::handle h = pybind11::cast(i);

    Foo f;

    pybind11::handle h2 = pybind11::cast(f);




    return h.ptr();
}
 



static PyObject *particle_getattro(PyObject* obj, PyObject *name) {
    
    PyObject *s = PyObject_Str(name);
    PyObject* pyStr = PyUnicode_AsEncodedString(s, "utf-8", "Error ~");
    const char *cstr = PyBytes_AS_STRING(pyStr);
    std::cout << obj->ob_type->tp_name << ": " << __PRETTY_FUNCTION__ << ":" << cstr << "\n";
    return PyObject_GenericGetAttr(obj, name);
}

static int particle_init(MxParticle *self, PyObject *args, PyObject *kwds) {
    std::cout << MX_FUNCTION << ", tp_name: " << self->ob_type->tp_name << std::endl;


    //self->charge = 1;
    //self->mass = 3.14;

    //PyObject *o = PyObject_GetAttrString(self, "mass");
    return 0;
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





PyGetSetDef particle_getsets[] = {
        //create_vector4_getset(),
    MakeAttibute("position", "doc", &MxParticle::position),
    MakeAttibute("velocity", "doc", &MxParticle::velocity),
    MakeAttibute("force", "doc", &MxParticle::force),
    MakeAttibute("q", "doc", &MxParticle::q),
    {NULL}
};



MxParticleType MxParticle_Type = {
{
  {
      PyVarObject_HEAD_INIT(NULL, 0)
      .tp_name =           "Particle",
      .tp_basicsize =      sizeof(MxParticle),
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
      .tp_new =            PyType_GenericNew,
      .tp_free =           0, 
      .tp_is_gc =          0, 
      .tp_bases =          0, 
      .tp_mro =            0, 
      .tp_cache =          0, 
      .tp_subclasses =     0, 
      .tp_weaklist =       0, 
      .tp_del =            0, 
      .tp_version_tag =    0, 
      .tp_finalize =       0    
    }
  }
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
    std::cout << MX_FUNCTION << std::endl;
    PyTypeObject *result;
    PyObject *fields;


    /* create the new instance (which is a class,
           since we are a metatype!) */
    result = (PyTypeObject *)PyType_Type.tp_new(type, args, kwds);
    //result = (PyTypeObject*)PyType_GenericNew(type, args, kwds);
    if (!result)
        return NULL;

    std::cout << "type->tp_name: " << type->tp_name << std::endl;

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
    MakeAttibute("mass", "doc", &MxParticleType::mass),
    {NULL},
};

static PyObject *
particle_type_descr_get(PyMemberDescrObject *descr, PyObject *obj, PyObject *type)
{
    return PyType_Type.tp_descr_get((PyObject*)descr, obj, type);
}

static int particle_type_init(MxParticleType *self, PyObject *args, PyObject *kwds) {
    std::cout << MX_FUNCTION << ", tp_name: " << self->ht_type.tp_name << std::endl;


    self->charge = 1;
    self->mass = 3.14;
    
    //PyObject *o = PyObject_GetAttrString(self, "mass");
    
    
    return 0;
}


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


/**
 * @brief Initialize a #part.
 *
 * @param p The #part to be initialized.
 * @param vid The virtual id of this #part.
 * @param type The numerical id of the particle type.
 * @param flags The particle flags.
 *
 * @return #part_err_ok or < 0 on error (see #part_err).
 *
 */

int md_particle_init ( struct MxParticle *p , int vid , int type , unsigned int flags ) {

    /* check inputs */
    if ( p == NULL )
        return particle_err = PARTICLE_ERR_NULL;

    /* Set the paticle data. */
    p->vid = vid;
    p->typeId = type;
    p->flags = flags;



    /* all is well... */
    return PARTICLE_ERR_OK;

}

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

HRESULT MxParticle_init(PyObject *m)
{
    
    
    
    f<>(&MxParticle::q);
    
    f<>(&MxParticle::position);
    
    f<>(&MxParticle::x);
    
    std::cout <<
    "sizeof PyHeapTypeObject: " << sizeof(PyHeapTypeObject) << ",\n"
    "sizeof MxParticleType: " << sizeof(MxParticleType) << ",\n"
    "offset of id: " << offsetof(MxParticleType, id) << ",\n"
    "offset of mass: " << offsetof(MxParticleType, mass) << ",\n"
    "offset of imass: " << offsetof(MxParticleType, imass) << ",\n"
    "offset of charge: " << offsetof(MxParticleType, charge) << ",\n"
    "offset of eps: " << offsetof(MxParticleType, eps) << ",\n"
    "offset of rmin: " << offsetof(MxParticleType, rmin) << ",\n"
    "offset of name: " << offsetof(MxParticleType, name) << ",\n"
    "offset of name2: " << offsetof(MxParticleType, name2) << "\n";

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
    
    PyMemberDef md =  {
            .name = "mass",
            .type = T_DOUBLE,
            .offset = offsetof(MxParticleType, mass),
            .flags = CDESCR_TYPE,
            .doc = NULL
    };

    PyObject *descr = CDescr_NewMember((PyTypeObject*)&MxParticle_Type, &md);

    if (descr == NULL)
        return -1;
    if (PyDict_SetItem(MxParticle_Type.ht_type.tp_dict, CDescr_NAME(descr), descr) < 0) {
        Py_DECREF(descr);
        return -1;
    }
    Py_DECREF(descr);



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

//static PyObject *
//PyCStructType_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
//{
//    return StructUnionType_new(type, args, kwds, 1);
//}
