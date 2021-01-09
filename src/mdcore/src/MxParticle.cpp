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
#define _USE_MATH_DEFINES // for C++

#include <MxParticle.h>

#include <cmath>
#include <stdlib.h>
#include "fptype.h"
#include <iostream>

// python type info
#include <structmember.h>
#include <MxNumpy.h>
#include <pybind11/pybind11.h>
#include <MxPy.h>
#include "engine.h"
#include "space.h"
#include "mx_runtime.h"

#include <MxParticleEvent.h>
#include "../../rendering/NOMStyle.hpp"
#include "MxCluster.hpp"
#include "metrics.h"
#include "MxConvert.hpp"
#include "MxParticleList.hpp"

#include <CSpecies.hpp>
#include <CSpeciesList.hpp>
#include <CStateVector.hpp>

#include <sstream>

struct Foo {
    int x; int y; int z;
};

static unsigned colors [] = {
    0xCCCCCC,
    0x6D99D3, // Rust Oleum Spa Blue
    0xF65917, // Rust Oleum Pumpkin
    0xF9CB20, // rust oleum yellow
    0x3CB371, // green
    0x6353BB, // SGI purple
    0xf4AC21, // gold
    0xC4A5DF, // light purple
    0xDC143C, // dark red
    0x1E90FF, // blue
    0xFFFF00, // yellow
    0x8A2BE2, // purple
    0x76D7C4, // light green
    0xF08080, // salmon
    0xFF00FF, // fuscia
    0xFF8C00, // orange
    0xFFDAB9, // tan / peach
    0x7F8C8D, // gray
    0x884EA0, // purple
    0x6B8E23,
    0x00FFFF,
    0xAFEEEE,
    0x008080,
    0xB0E0E6,
    0x6495ED,
    0x191970,
    0x0000CD,
    0xD8BFD8,
    0xFF1493,
    0xF0FFF0,
    0xFFFFF0,
    0xFFE4E1,
    0xDCDCDC,
    0x778899,
    0x000000
};

unsigned int *MxParticle_Colors = colors;


//template <typename C, typename D, typename... Extra>
//class_ &def_readwrite(const char *name, D C::*pm, const Extra&... extra) {

static int particle_init(MxParticleHandle *self, PyObject *_args, PyObject *_kwds);

static int particle_init_ex(MxParticleHandle *self,  const Magnum::Vector3 &position,
                            const Magnum::Vector3 &velocity,
                            int clusterId);
    
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

static void printTypeInfo(const char* name, PyTypeObject *p);

static PyObject* particle_destroy(MxParticleHandle *part, PyObject *args);

static PyObject* particle_spherical(MxParticleHandle *part, PyObject *args);

static PyObject* particle_fission(MxParticleHandle *part, PyObject *args, PyObject *kwargs);

static PyObject* particle_virial(MxParticleHandle *_self, PyObject *args, PyObject *kwargs);

static PyObject* particle_become(MxParticleHandle *_self, PyObject *args, PyObject *kwargs);

static PyObject* particle_distance(MxParticleHandle *_self, PyObject *args, PyObject *kwargs);

static PyObject* particle_neighbors(MxParticleHandle *_self, PyObject *args, PyObject *kwargs);

static PyObject* particletype_items(MxParticleType *self);

static PyObject *particle_repr(MxParticleHandle *obj);

//static PyObject *particle_getattro(PyObject* obj, PyObject *name) {
//
//    PyObject *s = PyObject_Str(name);
//    PyObject* pyStr = PyUnicode_AsEncodedString(s, "utf-8", "Error ~");
//    const char *cstr = PyBytes_AS_STRING(pyStr);
//    std::cout << obj->ob_type->tp_name << ": " << __PRETTY_FUNCTION__ << ":" << cstr << "\n";
//    return PyObject_GenericGetAttr(obj, name);
//}






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

            bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
            bool isParticleType = PyObject_IsInstance(obj, (PyObject*)&MxParticleType_Type);

            std::cout << "is particle: " << isParticle << std::endl;
            std::cout << "is particle type: " << isParticleType << std::endl;
            return PyLong_FromLong(567);
        },
        .set = [](PyObject *obj, PyObject *, void *p) -> int {
            const char* on = obj != NULL ? obj->ob_type->tp_name : "NULL";
            std::cout << "setter(obj.type:" << on << ", p:" << p << ")" << std::endl;

            bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
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
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->charge).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        
        try {
            double *x = &type->charge;
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
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->mass).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        
        try {
            double *x = &type->mass;
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

PyGetSetDef gs_frozen = {
    .name = "frozen",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        bool frozen = 0;
        if(isParticle) {
            MxParticleHandle *pobj = (MxParticleHandle*)obj;
            frozen = _Engine.s.partlist[pobj->id]->flags & PARTICLE_FROZEN;
        }
        else {
            MxParticleType *type = (MxParticleType*)obj;
            assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
            frozen = type->particle_flags & PARTICLE_FROZEN;
        }
        return pybind11::cast(frozen).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        
        try {
            if(isParticle) {
                MxParticle *part = ((MxParticleHandle*)obj)->part();
                bool b = val == Py_True;
                if(b) {
                    part->flags |= PARTICLE_FROZEN;
                }
                else {
                    part->flags &= ~PARTICLE_FROZEN;
                }
                return 0;
            }
            else {
                MxParticleType *type = (MxParticleType*)obj;
                bool b = val == Py_True;
                if(b) {
                    type->particle_flags |= PARTICLE_FROZEN;
                }
                else {
                    type->particle_flags &= ~PARTICLE_FROZEN;
                }
                return 0;
            }
        }
        catch (const pybind11::builtin_exception &e) {
            e.set_error();
            return -1;
        }
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_style = {
    .name = "style",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        NOMStyle *style = NULL;
        if(isParticle) {
            MxParticleHandle *pyp = (MxParticleHandle*)obj;
            style = _Engine.s.partlist[pyp->id]->style;
        }
        else {
            style =  ((MxParticleType*)obj)->style;
        }
        if(style) {
            Py_INCREF(style);
            return (PyObject*)style;
        }
        Py_RETURN_NONE;
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        PyErr_SetString(PyExc_PermissionError, "read only");
        return -1;
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_age = {
    .name = "age",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        MxParticleHandle *part = (MxParticleHandle*)obj;
        double age = _Engine.s.partlist[part->id]->creation_time * _Engine.dt;
        return pybind11::cast(age).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        PyErr_SetString(PyExc_PermissionError, "read only");
        return -1;
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_dynamics = {
    .name = "dynamics",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(MxParticleDynamics(type->dynamics)).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        
        try {
            type->dynamics = (unsigned char)pybind11::cast<MxParticleDynamics>(val);
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

PyGetSetDef gs_radius = {
    .name = "radius",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        float radius = 0;
        if(isParticle) {
            MxParticleHandle *pobj = (MxParticleHandle*)obj;
            radius = _Engine.s.partlist[pobj->id]->radius;
        }
        else {
            MxParticleType *type = (MxParticleType*)obj;
            assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
            radius = type->radius;
            
        }
        return pybind11::cast(radius).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        
        try {
            if(isParticle) {
                float *pradius = nullptr;
                MxParticleHandle *pobj = (MxParticleHandle*)obj;
                pradius = &(_Engine.s.partlist[pobj->id]->radius);
                *pradius = pybind11::cast<float>(val);
                return 0;
            }
            else {
                double *pradius;
                MxParticleType *type = (MxParticleType*)obj;
                assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
                pradius = &(type->radius);
                *pradius = pybind11::cast<double>(val);
                return 0;
            }
        }
        catch (const pybind11::builtin_exception &e) {
            e.set_error();
            return -1;
        }
    },
    .doc = "test doc",
    .closure = NULL
};

// only valid on type
PyGetSetDef gs_minimum_radius = {
    .name = "minimum_radius",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        MxParticleType *type = (MxParticleType*)obj;
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->minimum_radius).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        try {
            MxParticleType *type = (MxParticleType*)obj;
            assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
            double* pradius = &(type->minimum_radius);
            *pradius = pybind11::cast<double>(val);
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
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->name).release().ptr();
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
        bool isParticle = PyObject_IsInstance(obj, (PyObject*)MxParticle_GetType());
        MxParticleType *type = NULL;
        if(isParticle) {
            type = (MxParticleType*)obj->ob_type;
        }
        else {
            type = (MxParticleType*)obj;
        }
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->name2).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        PyErr_SetString(PyExc_PermissionError, "read only");
        return -1;
    },
    .doc = "test doc",
    .closure = NULL
};

// temperature is an ensemble property
PyGetSetDef gs_type_temperature = {
    .name = "temperature",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        MxParticleType *type = type = (MxParticleType*)obj;
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->kinetic_energy).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            return -1;
    },
    .doc = "test doc",
    .closure = NULL
};

PyGetSetDef gs_type_target_temperature = {
    .name = "target_temperature",
    .get = [](PyObject *obj, void *p) -> PyObject* {
        MxParticleType *type = (MxParticleType*)obj;
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        return pybind11::cast(type->target_energy).release().ptr();
    },
    .set = [](PyObject *obj, PyObject *val, void *p) -> int {
        MxParticleType *type = (MxParticleType*)obj;
        assert(type && PyObject_IsInstance((PyObject*)type, (PyObject*)&MxParticleType_Type));
        
        try {
            double *x = &type->target_energy;
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



PyGetSetDef particle_getsets[] = {
    gs_charge,
    gs_mass,
    gs_radius,
    gs_name,
    gs_name2,
    gs_dynamics,
    gsd,
    gs_age,
    gs_style,
    gs_frozen,
    {
        .name = "position",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            int id = ((MxParticleHandle*)obj)->id;
            Magnum::Vector3 vec;
            space_getpos(&_Engine.s, id, vec.data());
            return mx::cast(vec);
            
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxParticleHandle*)obj)->id;
                Magnum::Vector3 vec = mx::cast<Magnum::Vector3>(val);
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
            int id = ((MxParticleHandle*)obj)->id;
            Magnum::Vector3 *vec = &_Engine.s.partlist[id]->velocity;
            return mx::cast(*vec);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxParticleHandle*)obj)->id;
                Magnum::Vector3 *vec = &_Engine.s.partlist[id]->velocity;
                *vec = mx::cast<Magnum::Vector3>(val);
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
            int id = ((MxParticleHandle*)obj)->id;
            Magnum::Vector3 *vec = &_Engine.s.partlist[id]->force;
            return mx::cast(*vec);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxParticleHandle*)obj)->id;
                Magnum::Vector3 *vec = &_Engine.s.partlist[id]->force;
                *vec = mx::cast<Magnum::Vector3>(val);
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
            int x = ((MxParticleHandle*)obj)->id;
            return pybind11::cast(x).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                // TODO read only
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
            int id = ((MxParticleHandle*)obj)->id;
            int x = _Engine.s.partlist[id]->typeId;
            return pybind11::cast(x).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                // TODO read only
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
            int id = ((MxParticleHandle*)obj)->id;
            unsigned short x = _Engine.s.partlist[id]->flags;
            return pybind11::cast(x).release().ptr();
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                int id = ((MxParticleHandle*)obj)->id;
                unsigned short *x = &_Engine.s.partlist[id]->flags;
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
    {
        .name = "species",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxParticle *part = MxParticle_Get(obj);
            if(part->state_vector) {
                Py_INCREF(part->state_vector);
                return part->state_vector;
            }
            Py_RETURN_NONE;
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

static PyMethodDef particle_methods[] = {
        { "fission", (PyCFunction)particle_fission, METH_VARARGS, NULL },
        { "split", (PyCFunction)particle_fission, METH_VARARGS, NULL }, // alias name
        { "destroy", (PyCFunction)particle_destroy, METH_VARARGS, NULL },
        { "spherical", (PyCFunction)particle_spherical, METH_VARARGS, NULL },
        { "spherical_position", (PyCFunction)particle_spherical, METH_VARARGS, NULL },
        { "virial", (PyCFunction)particle_virial, METH_VARARGS | METH_KEYWORDS, NULL },
        { "become", (PyCFunction)particle_become, METH_VARARGS | METH_KEYWORDS, NULL },
        { "neighbors", (PyCFunction)particle_neighbors, METH_VARARGS | METH_KEYWORDS, NULL },
        { "distance", (PyCFunction)particle_distance, METH_VARARGS | METH_KEYWORDS, NULL },
        { NULL, NULL, 0, NULL }
};


static PyMethodDef particletype_methods[] = {
    { "items", (PyCFunction)particletype_items, METH_NOARGS, NULL },
    { NULL, NULL, 0, NULL }
};



static PyObject* particle_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    //std::cout << MX_FUNCTION << ", type: " << type->tp_name << std::endl;
    return PyType_GenericNew(type, args, kwargs);
}




/**
 * The base particle type
 * this instance points to the 0'th item in the global engine struct.
 *
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
};
*/



static getattrofunc savedFunc = NULL;

static PyObject *particle_type_getattro(PyObject* obj, PyObject *name) {
    
    // PyObject *s = PyObject_Str(name);
    // PyObject* pyStr = PyUnicode_AsEncodedString(s, "utf-8", "Error ~");
    //const char *cstr = PyBytes_AS_STRING(pyStr);
    //std::cout << obj->ob_type->tp_name << ": " << __PRETTY_FUNCTION__ << ":" << cstr << "\n";
    return savedFunc(obj, name);
}

/**
 * Basically a copy of Python's
 * PyType_GenericAlloc(PyTypeObject *type, Py_ssize_t nitems)
 *
 * Want to format the memory identically to Python, except we allocate
 * the object in the engine's static array of types.
 */
PyObject *particle_type_alloc(PyTypeObject *type, Py_ssize_t nitems)
{    
    assert(nitems == 0);
    assert(type->tp_basicsize == sizeof(MxParticleType));

    MxParticleType *obj;
    const size_t size = sizeof(MxParticleType);
    /* note that we need to add one, for the sentinel */

    if (PyType_IS_GC(type)) {
        PyErr_SetString(PyExc_MemoryError, "Fatal error, particle type can not be a garbage collected type");
        return NULL;
    }
    else if(engine::nr_types >= engine::max_type) {
        std::cout << "out of memory for new type " << engine::nr_types << std::endl;
        PyErr_SetString(PyExc_MemoryError, "out of memory for new particle type");
        return NULL;
    }

    std::cout << "Creating new particle type " << engine::nr_types << std::endl;

    obj = &engine::types[engine::nr_types];
    memset(obj, '\0', size);
    obj->id = engine::nr_types;
    engine::nr_types++;

    if (type->tp_flags & Py_TPFLAGS_HEAPTYPE)
        Py_INCREF(type);

    if (type->tp_itemsize == 0)
        (void)PyObject_INIT(obj, type);
    else
        (void) PyObject_INIT_VAR((PyVarObject *)obj, type, nitems);

    return (PyObject*)obj;
}

static PyObject *
particle_type_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    std::string t = pybind11::cast<pybind11::str>((PyObject*)type);
    std::string a = pybind11::cast<pybind11::str>(args);
    std::string k = pybind11::cast<pybind11::str>(kwds);
    
    std::cout << MX_FUNCTION << "(type: " << t << ", args: " << a << ", kwargs: " << k << ")" << std::endl;
    
    PyTypeObject *result;

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

    * Constant physical characteristics
    double mass, imass, charge;

    *Nonbonded interaction parameters.
    double eps, rmin;

    * Name of this paritcle type.
    char name[64], name2[64];
 */


static PyGetSetDef particle_type_getset[] = {
    gsd,
    gs_charge,
    gs_mass,
    gs_radius,
    gs_minimum_radius,
    gs_name,
    gs_name2,
    gs_dynamics,
    gs_type_temperature,
    gs_type_target_temperature,
    gs_style,
    gs_frozen,
    {
        .name = "species",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxParticleType *type = (MxParticleType*)obj;
            if(type->species) {
                Py_INCREF(type->species);
                return type->species;
            }
            Py_RETURN_NONE;
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_PermissionError, "read only");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL},
};

//static PyObject *
//particle_type_descr_get(PyMemberDescrObject *descr, PyObject *obj, PyObject *type)
//{
//    std::cout << "PyType_Type.tp_descr_get: " << PyType_Type.tp_descr_get << std::endl;
//    return PyType_Type.tp_descr_get((PyObject*)descr, obj, type);
//}

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
    .tp_methods =        particletype_methods,
    .tp_members =        0,
    .tp_getset =         particle_type_getset,
    .tp_base =           0, 
    .tp_dict =           0, 
    .tp_descr_get =      0,
    .tp_descr_set =      0, 
    .tp_dictoffset =     0, 
    .tp_init =           (initproc)particle_type_init,
    .tp_alloc =          particle_type_alloc, 
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



//static void printTypeInfo(const char* name, PyTypeObject *p) {
//
//    uint32_t is_gc = p->tp_flags & Py_TPFLAGS_HAVE_GC;
//
//
//    std::cout << "type: {" << std::endl;
//    std::cout << "  name: " << name << std::endl;
//    std::cout << "  type_name: " << Py_TYPE(p)->tp_name << std::endl;
//    std::cout << "  basetype_name:" << p->tp_base->tp_name << std::endl;
//    std::cout << "  have gc: " << std::to_string((bool)PyType_IS_GC(p)) << std::endl;
//    std::cout << "}" << std::endl;
//
//    /*
//    if(p->tp_getattro) {
//        PyObject *o = PyUnicode_FromString("foo");
//        p->tp_getattro((PyObject*)p, o);
//    }
//     */
//}

HRESULT _MxParticle_init(PyObject *m)
{
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
    
    // clear the GC of the particle type. PyTypeReady causes this to
    // inherit flags from the base type, PyType_Type. Because we
    // manage our own memory, clear these bits.
    MxParticleType_Type.tp_flags &= ~(Py_TPFLAGS_HAVE_GC);
    MxParticleType_Type.tp_clear = NULL;
    MxParticleType_Type.tp_traverse = NULL;
    

    /*************************************************
     *
     * Classes using a custom metaclass second
     */
    // Py_TYPE(&Struct_Type) = &PyCStructType_Type;
    // Struct_Type.tp_base = &PyCData_Type;
    // if (PyType_Ready(&Struct_Type) < 0)
    //     return NULL;
    //Py_TYPE(&MxParticle_Type) = &MxParticleType_Type;
    //MxParticle_Type.tp_base = &PyBaseObject_Type;
    //if (PyType_Ready((PyTypeObject*)&MxParticle_Type) < 0) {
    //    std::cout << "could not initialize MxParticle_Type " << std::endl;
    //    return E_FAIL;
    //}
    
    if(MxParticleType_Type.tp_getattro) {
        savedFunc = MxParticleType_Type.tp_getattro;
        MxParticleType_Type.tp_getattro = particle_type_getattro;
    }
    
    Py_INCREF(&MxParticleType_Type);
    if (PyModule_AddObject(m, "ParticleType", (PyObject *)&MxParticleType_Type) < 0) {
        Py_DECREF(&MxParticleType_Type);
        return E_FAIL;
    }

    //Py_INCREF(&MxParticle_Type);
    //if (PyModule_AddObject(m, "Particle", (PyObject *)&MxParticle_Type) < 0) {
    //    Py_DECREF(&MxParticle_Type);
    //    return E_FAIL;
    //}
    
    pybind11::enum_<MxParticleDynamics>(m, "Dynamics")
    .value("Newtonian", MxParticleDynamics::PARTICLE_NEWTONIAN)
    .value("Overdamped", MxParticleDynamics::PARTICLE_OVERDAMPED)
    .export_values();

    return  engine_particle_base_init(m);
}

int MxParticle_Check(PyObject *o)
{
    return o && PyObject_IsInstance(o, (PyObject*)MxParticle_GetType());
}


MxParticleType* MxParticleType_New(const char *_name, PyObject *dict)
{    
    pybind11::str name(_name);
    pybind11::tuple bases(1);
    bases[0] = (PyObject*)MxParticle_GetType();
    pybind11::tuple args(3);
    args[0] = name;
    args[1] = bases;
    args[2] = dict;

    MxParticleType *result = (MxParticleType*)PyType_Type.tp_call((PyObject*)&PyType_Type, args.ptr(), NULL);

    assert(result && PyType_IsSubtype((PyTypeObject*)result, (PyTypeObject*)MxParticle_GetType()));

    return result;
}

static HRESULT particletype_copy_base_descriptors(PyTypeObject *new_type, PyObject *base_dict) {
    PyObject *new_dict = new_type->tp_dict;
    PyObject *values = PyDict_Values(base_dict);
    int size = PyList_Size(values);
    
    for(int i = 0; i < size; ++i) {
        PyObject *o = PyList_GetItem(values, i);
        
        if(Py_TYPE(o) == &PyMethodDescr_Type) {
            PyMethodDescrObject *desc_obj = (PyMethodDescrObject*)o;
            
            if(PyDict_GetItem(new_dict, desc_obj->d_common.d_name) == NULL) {
                
                o = PyDescr_NewMethod(new_type, desc_obj->d_method);
                PyDict_SetItem(new_dict, desc_obj->d_common.d_name, o);
            }
        }
    }
    
    return S_OK;
}
    


HRESULT MxParticleType_Init(MxParticleType *self, PyObject *_dict)
{
    assert(self->ht_type.tp_base &&
           PyType_IsSubtype(self->ht_type.tp_base, (PyTypeObject*)&engine::types[0]));
    
    MxParticleType *base = (MxParticleType*)self->ht_type.tp_base;
    
    self->parts.init();
    self->mass = base->mass;
    self->charge = base->charge;
    self->target_energy = base->target_energy;
    self->radius = base->radius;
    self->dynamics = base->dynamics;
    self->minimum_radius = base->minimum_radius;
    self->type_flags = base->type_flags;
    self->particle_flags = base->particle_flags;

    std::strncpy(self->name, self->ht_type.tp_name, MxParticleType::MAX_NAME);
    
    try {
        pybind11::dict dict = pybind11::reinterpret_borrow<pybind11::dict>(_dict);
        if(dict.contains("mass")) {
            self->mass = dict["mass"].cast<double>();
        }
        
        if(dict.contains("target_temperature")) {
            self->target_energy = dict["target_temperature"].cast<double>();
        }

        self->imass = 1.0 / self->mass;

        if(dict.contains("charge")) {
            self->charge = dict["charge"].cast<double>();
        }
        
        if(dict.contains("radius")) {
            self->radius = dict["radius"].cast<double>();
        }
        
        if(dict.contains("minimum_radius")) {
            self->minimum_radius = dict["minimum_radius"].cast<double>();
        }
        
        if(dict.contains("name2")) {
            std::string name2 = dict["name2"].cast<std::string>();
            std::strncpy(self->name2, name2.c_str(), MxParticleType::MAX_NAME);
        }
        
        if(dict.contains("dynamics")) {
            self->dynamics = dict["dynamics"].cast<MxParticleDynamics>();
        }
        
        PyObject *frozen = PyDict_GetItemString(_dict, "frozen");
        if(frozen) {
            if(mx::check<bool>(frozen)) {
                if(frozen == Py_True) {
                    self->particle_flags |= PARTICLE_FROZEN;
                }
                else if(frozen == Py_False) {
                    self->particle_flags &= ~PARTICLE_FROZEN;
                }
            }
            else {
                Magnum::Vector3i frozen_vec = mx::cast<Magnum::Vector3i>(frozen);
                if(frozen_vec[0]) {
                    self->particle_flags |= PARTICLE_FROZEN_X;
                }
                else {
                    self->particle_flags &= ~PARTICLE_FROZEN_X;
                }
                if(frozen_vec[1]) {
                    self->particle_flags |= PARTICLE_FROZEN_Y;
                }
                else {
                    self->particle_flags &= ~PARTICLE_FROZEN_Y;
                }
                if(frozen_vec[2]) {
                    self->particle_flags |= PARTICLE_FROZEN_Z;
                }
                else {
                    self->particle_flags &= ~PARTICLE_FROZEN_Z;
                }
            }
        }
        
        if(dict.contains("species")) {
            self->species = CSpeciesList_NewFromPyArgs(dict["species"].ptr());
        }
        else {
            self->species = NULL;
        }
        
        if(dict.contains("style")) {
            self->style = NOMStyle_New((PyObject*)self, dict["style"].ptr());
        }
        else {
            // copy base class style
            self->style = NOMStyle_Clone(((MxParticleType*)self->ht_type.tp_base)->style);
            Py_INCREF(self->style);
            // cycle the colors
            // nr_types is one more that the type id, so dec by one.
            self->style->color = Magnum::Color3::fromSrgb(
                colors[(_Engine.nr_types - 1) % (sizeof(colors)/sizeof(unsigned))]);
        }
        
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
            key = pybind11::cast("radius");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("minimum_radius");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("name2");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("target_temperature");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("dynamics");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("style");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("frozen");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
            key = pybind11::cast("species");
            if(PyDict_Contains(_dict, key.ptr())) {
                PyDict_DelItem(_dict, key.ptr());
            }
        }
        
        particletype_copy_base_descriptors((PyTypeObject*)self, base->ht_type.tp_dict);
        
        
        /*
         * move these to make an event decorator.
        
        if(CDict_ContainsItemString(_dict, "events")) {
            MyParticleType_BindEvents(self, PyDict_GetItemString(_dict, "events"));
        }

        // bind all the events that are in the type dictionary
        MyParticleType_BindEvents(self, PyDict_Values(_dict));
         */
        
        // special stuff for cluster types
        if(PyType_IsSubtype(self->ht_type.tp_base, (PyTypeObject*)MxCluster_GetType())) {
            return MxClusterType_Init(self, _dict);
        }
        else {
            return S_OK;
        }
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

CAPI_FUNC(MxParticleType*) MxParticle_GetType()
{
    return &engine::types[0];
}

CAPI_FUNC(MxParticleType*) MxCluster_GetType()
{
    return &engine::types[1];
}

HRESULT engine_particle_base_init(PyObject *m)
{
    if(engine::max_type < 3) {
        return mx_error(E_FAIL, "must have at least space for 3 particle types");
    }

    if(engine::nr_types != 0) {
        return mx_error(E_FAIL, "engine types already set");
    }

    if((engine::types = (MxParticleData *)malloc( sizeof(MxParticleData) * engine::max_type )) == NULL ) {
        return mx_error(E_FAIL, "could not allocate types memory");
    }
    
    ::memset(engine::types, 0, sizeof(MxParticleData) * engine::max_type);

    //make an instance of the base particle type, all new instances of base
    //class mechanica.Particle will be of this type
    PyTypeObject *ob = (PyTypeObject*)&engine::types[0];
    
    Py_TYPE(ob) =          &MxParticleType_Type;
    ob->tp_base =          &PyBaseObject_Type;
    ob->tp_getset =        particle_getsets;
    ob->tp_methods =       particle_methods;
    ob->tp_name =          "Particle";
    ob->tp_basicsize =     sizeof(MxParticleHandle);
    ob->tp_flags =         Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    ob->tp_doc =           "Custom objects";
    ob->tp_init =          (initproc)particle_init;
    ob->tp_new =           particle_new;
    ob->tp_as_sequence =   &MxCluster_Sequence;
    ob->tp_del =           [] (PyObject *p) -> void {
        std::cout << "tp_del MxParticleHandle" << std::endl;
    };
    ob->tp_finalize =      [] (PyObject *p) -> void {
        std::cout << "tp_finalize MxParticleHandle" << std::endl;
    };
    ob->tp_str =           (reprfunc)particle_repr;
    ob->tp_repr =          (reprfunc)particle_repr;
    

    if(PyType_Ready(ob) < 0) {
        return mx_error(E_FAIL, "PyType_Ready on base particle failed");
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
    pt->style = NOMStyle_NewEx(Magnum::Color3::fromSrgb(colors[0]));

    ::strncpy(pt->name, "Particle", MxParticleType::MAX_NAME);
    ::strncpy(pt->name2, "Particle", MxParticleType::MAX_NAME);
    
    // set the singlton particle type data to the new item here.
    if (PyModule_AddObject(m, "Particle", (PyObject*)&engine::types[0]) < 0) {
        return E_FAIL;
    }

    std::cout << "added Particle to mechanica module" << std::endl;
    
    engine::nr_types = 1;
    
    return S_OK;
}

PyObject* particle_destroy(MxParticleHandle *part, PyObject *args)
{
    if(SUCCEEDED(engine_del_particle(&_Engine, part->id))) {
        Py_RETURN_NONE;
    }
    // c_error should set the python error
    return NULL;
}

PyObject* particle_spherical(MxParticleHandle *_self, PyObject *args)
{
    try {
        MxParticle *self = MxParticle_Get(_self);
        // c_error should set the python error
        
        Magnum::Vector3 origin;
        if(PyTuple_Check(args) && PyTuple_Size(args) > 0) {
            MxParticle *part = MxParticle_Get(PyTuple_GET_ITEM(args, 0));
            
            if(part) {
                origin = part->global_position();
            }
            else {
                origin = mx::cast<Magnum::Vector3>(PyTuple_GET_ITEM(args, 0));
            }
        }
        else {
            origin = Magnum::Vector3{
                (float)_Engine.s.dim[0],
                (float)_Engine.s.dim[1],
                (float)_Engine.s.dim[2]} / 2.;
        }
        return MPyCartesianToSpherical(self->global_position(), origin);
    }
    catch (const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

PyObject* particle_virial(MxParticleHandle *_self, PyObject *args, PyObject *kwargs)
{
    try {
        MxParticle *self = MxParticle_Get(_self);
        Magnum::Vector3 pos = self->global_position();
        Magnum::Matrix3 mat;
        
        float radius;
        
        if(PyTuple_Size(args) > 0) {
            radius = mx::cast<float>(PyTuple_GetItem(args, 0));
        }
        else {
            radius = self->radius * 10;
        }
        
        std::set<short int> typeIds;
        for(int i = 0; i < _Engine.nr_types; ++i) {
            typeIds.emplace(i);
        }
        
        HRESULT result = MxCalculateVirial(pos.data(), radius, typeIds, mat.data());
        
        return mx::cast(mat);
    }
    catch(const std::exception &e) {
        c_exp(e, "invalid args");
        return NULL;
    }
}

// reprfunc PyTypeObject.tp_repr
// An optional pointer to a function that implements the built-in function repr().
//
// The signature is the same as for PyObject_Repr():
//
// PyObject *tp_repr(PyObject *self);
// The function must return a string or a Unicode object. Ideally, this function should
// return a string that, when passed to eval(), given a suitable environment, returns an
// object with the same value. If this is not feasible, it should return a string starting
// with '<' and ending with '>' from which both the type and the value of the object
// can be deduced.
PyObject *particle_repr(MxParticleHandle *obj) {
    MxParticle *p = obj->part();
    MxParticleType *type = &_Engine.types[p->typeId];
    std::stringstream  ss;
    
    Magnum::Vector3 pos = p->global_position();
    
    ss << type->name << "(";
    ss << "id=" << p->id << ", ";
    ss << "position=[" << pos[0] << "," << pos[1] << "," << pos[2] << "]";
    ss << ")";
    
    return PyUnicode_FromString(ss.str().c_str());
}


PyObject *MxParticle_BasicFission(MxParticle *part) {
    Py_RETURN_NONE;

}

HRESULT MxParticleType::addpart(int32_t id)
{
    this->parts.insert(id);
    return S_OK;
}


/**
 * remove a particle id from this type
 */
HRESULT MxParticleType::del_part(int32_t id) {
    this->parts.remove(id);
    return S_OK;
}

PyObject* MxParticle_FissionSimple(MxParticle *self,
        MxParticleType *a, MxParticleType *b, int nPartitionRatios,
        float *partitionRations)
{
    int self_id = self->id;

    MxParticleType *type = &_Engine.types[self->typeId];
    
    // volume preserving radius
    float r2 = self->radius / std::pow(2., 1/3.);
    
    if(r2 < type->minimum_radius) {
        Py_RETURN_NONE;
    }

    MxParticle part = {};
    part.mass = self->mass;
    part.position = self->position;
    part.velocity = self->velocity;
    part.force = {};
    part.persistent_force = {};
    part.q = self->q;
    part.radius = self->radius;
    part.id = engine_next_partid(&_Engine);
    part.vid = 0;
    part.typeId = type->id;
    part.flags = self->flags;
    part._pyparticle = NULL;
    part.parts = NULL;
    part.nr_parts = 0;
    part.size_parts = 0;
    part.creation_time = _Engine.time;
    if(part.radius > _Engine.s.cutoff) {
        part.flags |= PARTICLE_LARGE;
    }

    std::uniform_real_distribution<float> x(-1, 1);

    Magnum::Vector3 sep = {x(CRandom), x(CRandom), x(CRandom)};
    sep = sep.normalized();
    sep = sep * r2;

    try {

        // create a new particle at the same location as the original particle.
        MxParticle *p = NULL;
        Magnum::Vector3 vec;
        space_getpos(&_Engine.s, self->id, vec.data());
        double pos[] = {vec[0], vec[1], vec[2]};
        int result = engine_addpart (&_Engine, &part, pos, &p);

        if(result < 0) {
            PyErr_SetString(PyExc_Exception, engine_err_msg[-engine_err]);
            return NULL;
        }
        
        // pointers after engine_addpart could change...
        self = _Engine.s.partlist[self_id];
        self->position += sep;
        
        // p is valid, because that's the result of the addpart
        p->position -= sep;
        
        // all is good, set the new radii
        self->radius = r2;
        p->radius = r2;
        self->mass = p->mass = self->mass / 2.;
        self->imass = p->imass = 1. / self->mass;

        return p->py_particle();
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return NULL;
    }
}

PyObject* particle_fission(MxParticleHandle *part, PyObject *args,
        PyObject *kwargs)
{
    try {
        //double min = arg<double>("min", 0, args, _kwargs);
        //double max = arg<double>("max", 1, args, _kwargs);
        //double A = arg<double>("A", 2, _args, _kwargs);
        //double B = arg<double>("B", 3, _args, _kwargs);
        //double tol = arg<double>("tol", 4, _args, _kwargs, 0.001 * (max-min));
        //return potential_create_LJ126( min, max, A, B, tol);
        

        assert(part->id == _Engine.s.partlist[part->id]->id);        
        MxParticle *p = _Engine.s.partlist[part->id];
        return MxParticle_FissionSimple(p, NULL, NULL, 0, NULL);
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return NULL;
    }
}

int MxParticleType_Check(PyObject* obj) {
    if(PyType_Check(obj)) {
        PyTypeObject *ptype = (PyTypeObject*)MxParticle_GetType();
        return PyType_IsSubtype((PyTypeObject *)obj, ptype);
    }
    return 0;
}

MxParticle* MxParticle_Get(PyObject* obj) {
    if(obj && MxParticle_Check(obj)) {
        MxParticleHandle *pypart = (MxParticleHandle*)obj;
        return _Engine.s.partlist[pypart->id];
    }
    return NULL;
}

MxParticleHandle *MxParticle::py_particle() {
    
    if(!this->_pyparticle) {

        PyTypeObject *type = (PyTypeObject*)&_Engine.types[this->typeId];
        MxParticleHandle *part = (MxParticleHandle*)PyType_GenericAlloc(type, 0);
        part->id = this->id;
        this->_pyparticle = part;
    }
    
    Py_INCREF(this->_pyparticle);
    return this->_pyparticle;
}


HRESULT MxParticle::addpart(int32_t pid) {
    
    /* do we need to extend the partlist? */
    if ( nr_parts == size_parts ) {
        size_parts += CLUSTER_PARTLIST_INCR;
        int32_t* temp;
        if ( ( temp = (int32_t*)malloc( sizeof(int32_t) * size_parts ) ) == NULL )
            return c_error(E_FAIL, "could not allocate space for type particles");
        memcpy( temp , parts , sizeof(int32_t) * nr_parts );
        free( parts );
        parts = temp;
    }
    
    MxParticle *p = _Engine.s.partlist[pid];
    p->clusterId = this->id;
    
    parts[nr_parts] = pid;
    nr_parts++;
    return S_OK;
}

HRESULT MxParticle::removepart(int32_t pid) {
    
    int pid_index = -1;
    
    for(int i = 0; i < this->nr_parts; ++i) {
        if(this->particle(i)->id == pid) {
            pid_index = i;
            break;
        }
    }
    
    if(pid_index < 0) {
        return mx_error(E_FAIL, "particle id not in this cluster");
    }
    
    MxParticle *p = _Engine.s.partlist[pid];
    p->clusterId = -1;
    
    for(int i = pid_index; i + 1 < this->nr_parts; ++i) {
        this->parts[i] = this->parts[i+1];
    }
    nr_parts--;
    
    return S_OK;
}

PyObject* MxParticle_New(PyObject *type, PyObject *args, PyObject *kwargs) {
    
    if(!PyType_Check(type)) {
        return NULL;
    }
    
    if(!PyObject_IsSubclass(type, (PyObject*)MxParticle_GetType())) {
        return NULL;
    }
    
    // make a new pyparticle
    PyObject *pyPart = PyType_GenericNew((PyTypeObject*)type, args, kwargs);
    
    if(!args) {
        args = PyTuple_New(0);
    }
    else{
        Py_INCREF(args);
    }
    
    if(!kwargs) {
        kwargs = PyDict_New();
    }
    else {
        Py_INCREF(kwargs);
    }
    
    
    if(particle_init((MxParticleHandle*)pyPart, args, kwargs) < 0) {
        std::cout << "bad stuff" << std::endl;
        return NULL;
    }
    
    Py_DECREF(args);
    Py_DECREF(kwargs);
    
    return pyPart;
}


Magnum::Vector3 MxRandomVector(float mean, float std) {
    std::normal_distribution<> dist{mean,std};
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    float theta = 2 * M_PI * uniform01(CRandom);
    float phi = acos(1 - 2 * uniform01(CRandom));
    float r = dist(CRandom);
    float x = r * sin(phi) * cos(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(phi);
    return Magnum::Vector3{x, y, z};
}

Magnum::Vector3 MxRandomUnitVector() {
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    float theta = 2 * M_PI * uniform01(CRandom);
    float phi = acos(1 - 2 * uniform01(CRandom));
    float r = 1.;
    float x = r * sin(phi) * cos(theta);
    float y = r * sin(phi) * sin(theta);
    float z = r * cos(phi);
    return Magnum::Vector3{x, y, z};
}


HRESULT MxParticle_Become(MxParticle *part, MxParticleType *type) {
    HRESULT hr;
    if(!part || !type) {
        return c_error(E_FAIL, "null arguments");
    }
    MxParticleHandle *pypart = part->py_particle();
    
    MxParticleType *currentType = &_Engine.types[part->typeId];
    
    assert(pypart->ob_type == (PyTypeObject*)currentType);
    
    if(!SUCCEEDED(hr = currentType->del_part(part->id))) {
        return hr;
    };
    
    if(!SUCCEEDED(hr = type->addpart(part->id))) {
        return hr;
    }
    
    pypart->ob_type = (PyTypeObject*)type;
    Py_DECREF(currentType);
    Py_INCREF(type);
    
    part->typeId = type->id;
    
    if(part->state_vector) {
        CStateVector *oldState = part->state_vector;
        
        if(type->species) {
            part->state_vector = CStateVector_New(type->species, pypart, oldState, 0, 0, 0);
        }
        else {
            part->state_vector = NULL;
        }
        
        Py_DECREF(oldState);
    }
    
    assert(type == &_Engine.types[part->typeId]);
    
    // TODO: bad things will happen if we convert between cluster and atomic types.
    
    return S_OK;
}

static PyObject* particle_become(MxParticleHandle *_self, PyObject *args, PyObject *kwargs) {
    if(args && PyTuple_Size(args) > 0) {
        MxParticleType *o = MxParticleType_Get(PyTuple_GetItem(args, 0));
        if(!o) {
            PyErr_SetString(PyExc_TypeError, "argument 0 is not a particle derived type");
            return NULL;
        }
        
        HRESULT hr;
        if(!SUCCEEDED((hr = MxParticle_Become(MxParticle_Get(_self), o)))) {
            c_error(hr, "could not convert particle type");
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

/**
 * checks of the given python object is a sequence of some sort,
 * and checks that every element is a MxParticleType, fills the
 * result set.
 *
 * Every object in the given object must be a type, otherwise
 * returns an empty set.
 */
HRESULT MxParticleType_IdsFromPythonObj(PyObject *obj, std::set<short int>& ids) {
    if(obj == NULL) {
        // get the type ids, we're a particle, so only select other particles
        // by default
        for(int i = 0; i < _Engine.nr_types; ++i) {
            if(PyType_IsSubtype((PyTypeObject*)&_Engine.types[i], (PyTypeObject*)MxCluster_GetType())) {
                continue;
            }
            ids.insert(i);
        }
        return S_OK;
    }
    
    if(PySequence_Check(obj)) {
        int len = PySequence_Length(obj);
        for(int i = 0; i < len; ++i) {
            PyObject *o = PySequence_GetItem(obj, i);
            if(MxParticleType_Check(o)) {
                MxParticleType *type = (MxParticleType*)o;
                ids.insert(type->id);
            }
            else {
                return E_FAIL;
            }
        }
        return S_OK;
    }
    
    else if(MxParticleType_Check(obj)) {
        MxParticleType *type = (MxParticleType*)obj;
        ids.insert(type->id);
        return S_OK;
    }
    
    return E_FAIL;
}

static PyObject* particle_neighbors(MxParticleHandle *_self, PyObject *args, PyObject *kwargs) {
    try {
        float radius;
        PyObject *_radius = mx::arg("distance", 0, args, kwargs);
        if(_radius) {
            radius = mx::cast<float>(_radius);
        }
        else {
            radius = _Engine.s.cutoff;
        }
        
        PyObject *ptypes = mx::arg("types", 1, args, kwargs);
        
        std::set<short int> types;
   
        if(FAILED(MxParticleType_IdsFromPythonObj(ptypes, types))) {
            throw std::invalid_argument("types must be a tuple, or a Particle derived type");
        }

        MxParticle *self = MxParticle_Get(_self);
        
        // take into account the radius of this particle.
        radius += self->radius;
        
        uint16_t nr_parts = 0;
        int32_t *parts = NULL;
        
        MxParticle_Neighbors(self, radius, &types, &nr_parts, &parts);
        
        return (PyObject*)MxParticleList_NewFromData(nr_parts, parts);
    }
    catch(std::exception &e) {
        c_exp(e, "error getting args");
        return NULL;
    }
}


static PyObject* particletype_items(MxParticleType *self) {
    PyObject *result = &self->parts;
    Py_INCREF(result);
    return result;
}

static PyObject* particle_distance(MxParticleHandle *_self, PyObject *args, PyObject *kwargs) {
    MxParticle *self = MxParticle_Get(_self);
    MxParticle *other = NULL;
    
    if(args && PyTuple_Size(args) > 0) {
        other = MxParticle_Get(PyTuple_GetItem(args, 0));
    }
    
    if(other == NULL || self == NULL) {
        c_error(E_FAIL, "invalid args, distance(Particle)");
        return NULL;
    }
    
    Magnum::Vector3 pos = self->global_position();
    Magnum::Vector3 opos = other->global_position();
    float d = (opos - pos).length();
    return PyFloat_FromDouble(d);
}


int particle_init(MxParticleHandle *self, PyObject *_args, PyObject *_kwds) {
    
    try {
        MxParticleType *type = (MxParticleType*)self->ob_type;
        
        pybind11::detail::loader_life_support ls{};
        pybind11::args args = pybind11::reinterpret_borrow<pybind11::args>(_args);
        pybind11::kwargs kwargs = pybind11::reinterpret_borrow<pybind11::kwargs>(_kwds);
        
        // make a random initial position
        std::uniform_real_distribution<float> x(_Engine.s.origin[0], _Engine.s.dim[0]);
        std::uniform_real_distribution<float> y(_Engine.s.origin[1], _Engine.s.dim[1]);
        std::uniform_real_distribution<float> z(_Engine.s.origin[2], _Engine.s.dim[2]);
        Magnum::Vector3 iniPos = {x(CRandom), y(CRandom), z(CRandom)};
        
        // initial velocity, chosen to fit target temperature
        std::uniform_real_distribution<float> v(-1.0, 1.0);
        Magnum::Vector3 vel = {v(CRandom), v(CRandom), v(CRandom)};
        float v2 = Magnum::Math::dot(vel, vel);
        float x2 = (type->target_energy * 2. / (type->mass * v2));
        vel *= std::sqrt(x2);
        
        Magnum::Vector3 position = arg<Magnum::Vector3>("position", 0, args.ptr(), kwargs.ptr(), iniPos);
        Magnum::Vector3 velocity = arg<Magnum::Vector3>("velocity", 1, args.ptr(), kwargs.ptr(), vel);
        
        // particle_init_ex will allocate a new particle, this can re-assign the pointers in
        // the engine particles, so need to pass cluster by id.
        MxParticle *cluster = _kwds  ? MxParticle_Get(PyDict_GetItemString(_kwds, "cluster")) : NULL;
        int clusterId = cluster ? cluster->id : -1;
        
        return particle_init_ex(self, position, velocity, clusterId);
        
    }
    catch (const pybind11::builtin_exception &e) {
        e.set_error();
        return -1;
    }
}

int particle_init_ex(MxParticleHandle *self,  const Magnum::Vector3 &position,
                     const Magnum::Vector3 &velocity,
                     int clusterId) {
    
    MxParticleType *type = (MxParticleType*)self->ob_type;
    
    MxParticle part;
    bzero(&part, sizeof(MxParticle));
    part.radius = type->radius;
    part.mass = type->mass;
    part.imass = type->imass;
    part.id = engine_next_partid(&_Engine);
    part.typeId = type->id;
    part.flags = type->particle_flags;
    part.creation_time = _Engine.time;
    part.clusterId = clusterId;
    
    if(type->species) {
        part.state_vector = CStateVector_New(type->species, self, NULL, 0, 0, 0);
    }
    
    if(PyObject_IsSubclass((PyObject*)type, (PyObject*)MxCluster_GetType())) {
        std::cout << "making cluster" << std::endl;
        part.flags |= PARTICLE_CLUSTER;
    }
    
    part.position = position;
    part.velocity = velocity;
    
    if(part.radius > _Engine.s.cutoff) {
        part.flags |= PARTICLE_LARGE;
    }
    
    MxParticle *p = NULL;
    double pos[] = {part.position[0], part.position[1], part.position[2]};
    int result = engine_addpart (&_Engine, &part, pos, &p);
    
    if(result < 0) {
        PyErr_SetString(PyExc_Exception, engine_err_msg[-engine_err]);
        return result;
    }
    
    self->id = p->id;
    
    
    if(clusterId >= 0) {
        MxParticle *cluster = _Engine.s.partlist[clusterId];
        p->flags |= PARTICLE_BOUND;
        cluster->addpart(p->id);
    } else {
        p->clusterId = -1;
    }
    
    Py_INCREF(self);
    p->_pyparticle = self;
    
    return 0;
}


MxParticleHandle* MxParticle_NewEx(PyObject *type,
                           const Magnum::Vector3 &pos, const Magnum::Vector3 &velocity,
                           int clusterId) {
    
    if(!PyType_Check(type)) {
        return NULL;
    }
    
    if(!PyObject_IsSubclass(type, (PyObject*)MxParticle_GetType())) {
        return NULL;
    }
    
    // make a new pyparticle
    MxParticleHandle *pyPart = (MxParticleHandle*)PyType_GenericNew((PyTypeObject*)type, NULL, NULL);
    

    if(particle_init_ex((MxParticleHandle*)pyPart, pos, velocity, clusterId) < 0) {
        std::cout << "bad stuff" << std::endl;
        return NULL;
    }

    return pyPart;
}


MxParticleType* MxParticleType_FindFromName(const char* name) {
    for(int i = 0; i < _Engine.nr_types; ++i) {
        MxParticleType *type = &_Engine.types[i];
        if(std::strncmp(name, type->name, sizeof(MxParticleType::name)) == 0) {
            return type;
        }
    }
    return NULL;
}




