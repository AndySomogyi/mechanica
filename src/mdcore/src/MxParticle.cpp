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



    self->pos = PyList_New(3);

    PyList_SetItem(self->pos, 0, PyFloat_FromDouble(0.0));
    PyList_SetItem(self->pos, 1, PyFloat_FromDouble(0.0));
    PyList_SetItem(self->pos, 2, PyFloat_FromDouble(0.0));

    self->vel = PyList_New(3);

    PyList_SetItem(self->vel, 0, PyFloat_FromDouble(0.0));
    PyList_SetItem(self->vel, 1, PyFloat_FromDouble(0.0));
    PyList_SetItem(self->vel, 2, PyFloat_FromDouble(0.0));

    self->force = PyList_New(3);

    PyList_SetItem(self->force, 0, PyFloat_FromDouble(0.0));
    PyList_SetItem(self->force, 1, PyFloat_FromDouble(0.0));
    PyList_SetItem(self->force, 2, PyFloat_FromDouble(0.0));


    return 0;
}

static PyMemberDef particle_members[] = {
    {
        .name = "position",
        .type = T_OBJECT,
        .offset = offsetof(MxParticle, pos),
        .flags = 0,
        .doc = NULL
    },
    {
        .name = "velocity",
        .type = T_OBJECT,
        .offset = offsetof(MxParticle, vel),
        .flags = 0,
        .doc = NULL
    },
    {
        .name = "force",
        .type = T_OBJECT,
        .offset = offsetof(MxParticle, force),
        .flags = 0,
        .doc = NULL
    },
    {NULL},
};




MxParticleType MxParticle_Type = {
        {
            {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "Particle",
            .tp_doc = "Custom objects",
            .tp_basicsize = sizeof(MxParticle),
            .tp_itemsize = 0,
            .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
            .tp_new = PyType_GenericNew,
            .tp_getattro = particle_getattro,
            .tp_init = (initproc)particle_init,
            .tp_members = particle_members
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


/*
 const char *name;
 int type;
 Py_ssize_t offset;
 int flags;
 const char *doc;
 */
static PyMemberDef particle_type_members[] = {
    {
        .name = "mass",
        .type = T_DOUBLE,
        .offset = offsetof(MxParticleType, mass),
        .flags = 0,
        .doc = NULL
    },
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
        .tp_name = "ParticleType",
        .tp_doc = "Custom objects",
        .tp_basicsize = sizeof(MxParticleType),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = particle_type_new,
        .tp_members = particle_type_members,
        .tp_descr_get = (descrgetfunc)particle_type_descr_get,
        .tp_init = (initproc)particle_type_init
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
    
    //PyMember_GetOne(NULL, NULL);
    
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
