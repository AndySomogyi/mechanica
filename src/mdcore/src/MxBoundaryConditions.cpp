/*
 * BoundaryConditions.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: andy
 */

#include "MxBoundaryConditions.hpp"
#include "MxConvert.hpp"
#include "space.h"
#include "engine.h"

#include <algorithm>
#include <string>


static PyObject* bc_str(MxBoundaryCondition *bc);

static PyObject* bcs_str(MxBoundaryConditions *bcs);

static PyGetSetDef bc_getset[] = {
    {
        .name = "name",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
            return mx::cast(std::string(bc->name));
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "kind",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
            std::string kind;
            switch(bc->kind) {
                case BOUNDARY_PERIODIC:
                    kind = "PERIODIC";
                    break;
                case BOUNDARY_FREESLIP:
                    kind = "FREESLIP";
                    break;
                case BOUNDARY_VELOCITY:
                    kind = "VELOCITY";
                    break;
                default:
                    kind = "INVALID";
                    break;
            }
            return mx::cast(kind);
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "velocity",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
            return mx::cast(bc->velocity);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
                bc->velocity = mx::cast<Magnum::Vector3>(val);
                return 0;
            }
            catch(const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "restore",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
            return mx::cast(bc->restore);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            try {
                MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
                bc->restore = mx::cast<float>(val);
                return 0;
            }
            catch(const std::exception &e) {
                return C_EXP(e);
            }
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

static PyGetSetDef bcs_getset[] = {
    {
        .name = "left",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryConditions *bcs = (MxBoundaryConditions*)obj;
            PyObject *res = &bcs->left;
            Py_INCREF(res);
            return res;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "right",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryConditions *bcs = (MxBoundaryConditions*)obj;
            PyObject *res = &bcs->right;
            Py_INCREF(res);
            return res;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "front",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryConditions *bcs = (MxBoundaryConditions*)obj;
            PyObject *res = &bcs->front;
            Py_INCREF(res);
            return res;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "back",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryConditions *bcs = (MxBoundaryConditions*)obj;
            PyObject *res = &bcs->back;
            Py_INCREF(res);
            return res;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "bottom",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryConditions *bcs = (MxBoundaryConditions*)obj;
            PyObject *res = &bcs->bottom;
            Py_INCREF(res);
            return res;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {
        .name = "top",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryConditions *bcs = (MxBoundaryConditions*)obj;
            PyObject *res = &bcs->top;
            Py_INCREF(res);
            return res;
        },
        .set = [](PyObject *_obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
        },
        .doc = "test doc",
        .closure = NULL
    },
    {NULL}
};

PyTypeObject MxBoundaryConditions_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "BoundaryConditions",
    .tp_basicsize =      sizeof(MxBoundaryConditions),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           (reprfunc)bcs_str,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            (reprfunc)bcs_str,
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
    .tp_getset =         bcs_getset,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)0,
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



PyTypeObject MxBoundaryCondition_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name =           "BoundaryCondition",
    .tp_basicsize =      sizeof(MxBoundaryCondition),
    .tp_itemsize =       0,
    .tp_dealloc =        0,
                         0, // .tp_print changed to tp_vectorcall_offset in python 3.8
    .tp_getattr =        0,
    .tp_setattr =        0,
    .tp_as_async =       0,
    .tp_repr =           (reprfunc)bc_str,
    .tp_as_number =      0,
    .tp_as_sequence =    0,
    .tp_as_mapping =     0,
    .tp_hash =           0,
    .tp_call =           0,
    .tp_str =            (reprfunc)bc_str,
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
    .tp_getset =         bc_getset,
    .tp_base =           0,
    .tp_dict =           0,
    .tp_descr_get =      0,
    .tp_descr_set =      0,
    .tp_dictoffset =     0,
    .tp_init =           (initproc)0,
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

/**
 * boundary was initialized from flags, set individual values
 */
static void boundaries_from_flags(MxBoundaryConditions *bc) {
    
    if(bc->periodic & space_periodic_x) {
        bc->left.kind = BOUNDARY_PERIODIC;
        bc->right.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_X) {
        bc->left.kind = BOUNDARY_FREESLIP;
        bc->right.kind = BOUNDARY_FREESLIP;
    }
    
    if(bc->periodic & space_periodic_y) {
        bc->front.kind = BOUNDARY_PERIODIC;
        bc->back.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_Y) {
        bc->front.kind = BOUNDARY_FREESLIP;
        bc->back.kind = BOUNDARY_FREESLIP;
    }
    
    if(bc->periodic & space_periodic_z) {
        bc->top.kind = BOUNDARY_PERIODIC;
        bc->bottom.kind = BOUNDARY_PERIODIC;
    }
    else if(bc->periodic & SPACE_FREESLIP_Z) {
        bc->top.kind = BOUNDARY_FREESLIP;
        bc->bottom.kind = BOUNDARY_FREESLIP;
    }
}

// check if object is string. If not string, return -1, if string,
// check if valid type, and return, if string and invalid string, throw exception.
static unsigned bc_kind_from_pystring(PyObject *o) {
    
    int result = -1;
    
    if(mx::check<std::string>(o)) {
        std::string s = mx::cast<std::string>(o);
        std::transform(s.begin(), s.end(),s.begin(), ::toupper);
        
        if(s.compare("PERIODIC") == 0) {
            result = BOUNDARY_PERIODIC;
        }
        
        else if (s.compare("FREE_SLIP") == 0 || s.compare("FREESLIP") == 0) {
            result = BOUNDARY_FREESLIP;
        }
        
        else if (s.compare("NO_SLIP") == 0 || s.compare("NOSLIP") == 0) {
            result = BOUNDARY_NO_SLIP;
        }
        
        else {
            std::string msg = "invalid choice of value for boundary condition, \"";
            msg += mx::cast<std::string>(o);
            msg += "\", only \"periodic\" and \"free_slip\" are supported for cardinal direction init";
            throw std::invalid_argument(msg);
        }
    }
    
    return result;
}

static int init_bc_direction(MxBoundaryCondition *low_bl, MxBoundaryCondition *high_bl, PyObject *o) {
    if(mx::check<std::string>(o)) {
        int kind = bc_kind_from_pystring(o);
        
        if(kind == BOUNDARY_NO_SLIP) {
            low_bl->kind = high_bl->kind = BOUNDARY_VELOCITY;
            low_bl->velocity = high_bl->velocity = Magnum::Vector3{0.f, 0.f, 0.f};
        }
        else {
            low_bl->kind = (BoundaryConditionKind)kind;
            high_bl->kind = (BoundaryConditionKind)kind;
        }
        
        return kind;
    }
    else {
        std::string msg = "invalid boundary initialization type (";
        msg += o->ob_type->tp_name;
        msg += "), only string and dictionary types are supported";
        throw std::invalid_argument(msg);
    }
}

// init a bc from a dict, dict is already known to be a dictionary
static unsigned init_bc(MxBoundaryCondition* bc, PyObject *o) {
    
    PyObject *item = PyDict_GetItemString(o, bc->name);
    
    if(!item) {
        return 0;
    }
    if(mx::check<std::string>(item)) {
        bc->kind = (BoundaryConditionKind)bc_kind_from_pystring(item);
        if(bc->kind == BOUNDARY_NO_SLIP) {
            bc->kind = BOUNDARY_VELOCITY;
            bc->velocity = Magnum::Vector3{};
        }
        return bc->kind;
    }
    else if(PyDict_Check(item)) {
        
        PyObject *vel = PyDict_GetItemString(item, "velocity");
        
        if(!vel) {
            throw std::invalid_argument("attempt to initialize a boundary condition with a "
                                        "dictionary that does not contain a \'velocity\' item, "
                                        "only velocity boundary conditions support dictionary init");
        }
        
        PyObject *r = PyDict_GetItemString(item, "restore");
        if(r) {
            bc->restore = mx::cast<float>(r);
        }
        
        bc->kind = BOUNDARY_VELOCITY;
        bc->velocity = mx::cast<Magnum::Vector3>(vel);
        return bc->kind;
    }
    else {
        std::string msg = "invalid type (";
        msg += item->ob_type->tp_name;
        msg += "), for \'";
        msg += bc->name;
        msg += "\', boundary condition initialization, only dictionary is supported";
        throw std::invalid_argument(msg);
    }
}

static void check_periodicy(MxBoundaryCondition *low_bc, MxBoundaryCondition *high_bc) {
    if((low_bc->kind & BOUNDARY_PERIODIC) ^ (high_bc->kind & BOUNDARY_PERIODIC)) {
        MxBoundaryCondition *has;
        MxBoundaryCondition *notHas;
    
        if(low_bc->kind & BOUNDARY_PERIODIC) {
            has = low_bc;
            notHas = high_bc;
        }
        else {
            has = high_bc;
            notHas = low_bc;
        }
        
        std::string msg = "only ";
        msg += has->name;
        msg += "has periodic boundary conditions set, but not ";
        msg += notHas->name;
        msg += ", setting both to periodic";
        
        low_bc->kind = BOUNDARY_PERIODIC;
        high_bc->kind = BOUNDARY_PERIODIC;
        
        PyErr_WarnEx(NULL, msg.c_str(), 0);
    }
}


HRESULT MxBoundaryConditions_Init(MxBoundaryConditions *bc, int *cells, PyObject *args) {
    
    if (PyType_IS_GC(&MxBoundaryConditions_Type)) {
        assert(0 && "should not get here");
    }
    
    bzero(bc, MxBoundaryConditions_Type.tp_basicsize);
    
    PyObject_INIT(bc, &MxBoundaryConditions_Type);
    PyObject_INIT(&(bc->top), &MxBoundaryCondition_Type);
    PyObject_INIT(&(bc->bottom), &MxBoundaryCondition_Type);
    PyObject_INIT(&(bc->left), &MxBoundaryCondition_Type);
    PyObject_INIT(&(bc->right), &MxBoundaryCondition_Type);
    PyObject_INIT(&(bc->front), &MxBoundaryCondition_Type);
    PyObject_INIT(&(bc->back), &MxBoundaryCondition_Type);
    
    bc->potenntials = (MxPotential**)malloc(6 * engine::max_type * sizeof(MxPotential**));
    
    bc->left.name = "left";     bc->left.restore = 1.f;     bc->left.potenntials =   &bc->potenntials[0 * engine::max_type];
    bc->right.name = "right";   bc->right.restore = 1.f;    bc->right.potenntials =  &bc->potenntials[1 * engine::max_type];
    bc->front.name = "front";   bc->front.restore = 1.f;    bc->front.potenntials =  &bc->potenntials[2 * engine::max_type];
    bc->back.name = "back";     bc->back.restore = 1.f;     bc->back.potenntials =   &bc->potenntials[3 * engine::max_type];
    bc->top.name = "top";       bc->top.restore = 1.f;      bc->top.potenntials =    &bc->potenntials[4 * engine::max_type];
    bc->bottom.name = "bottom"; bc->bottom.restore = 1.f;   bc->bottom.potenntials = &bc->potenntials[5 * engine::max_type];
    
    if(args) {
        try {
            if(mx::check<int>(args)) {
                bc->periodic = mx::cast<uint32_t>(args);
                boundaries_from_flags(bc);
            }
            else if(PyDict_Check(args)) {
                // check if we have x, y, z directions
                PyObject *o;
                
                if((o = PyDict_GetItemString(args, "x"))) {
                    switch(init_bc_direction(&(bc->left), &(bc->right), o)) {
                        case BOUNDARY_PERIODIC:
                            bc->periodic |= space_periodic_x;
                            break;
                        case BOUNDARY_FREESLIP:
                            bc->periodic |= SPACE_FREESLIP_X;
                            break;
                        default:
                            assert(0);
                    }
                }
                
                if((o = PyDict_GetItemString(args, "y"))) {
                    switch(init_bc_direction(&(bc->front), &(bc->back), o)) {
                        case BOUNDARY_PERIODIC:
                            bc->periodic |= space_periodic_y;
                            break;
                        case BOUNDARY_FREESLIP:
                            bc->periodic |= SPACE_FREESLIP_Y;
                            break;
                        default:
                            assert(0);
                    }
                }
                
                if((o = PyDict_GetItemString(args, "z"))) {
                    switch(init_bc_direction(&(bc->top), &(bc->bottom), o)) {
                        case BOUNDARY_PERIODIC:
                            bc->periodic |= space_periodic_z;
                            break;
                        case BOUNDARY_FREESLIP:
                            bc->periodic |= SPACE_FREESLIP_Z;
                            break;
                        default:
                            assert(0);
                    }
                }
                
                switch(init_bc(&(bc->left), args)) {
                    case BOUNDARY_PERIODIC:
                        bc->periodic |= space_periodic_x;
                        break;
                    case BOUNDARY_FREESLIP:
                        bc->periodic |= SPACE_FREESLIP_X;
                        break;
                }
                
                switch(init_bc(&(bc->right), args)) {
                    case BOUNDARY_PERIODIC:
                        bc->periodic |= space_periodic_x;
                        break;
                    case BOUNDARY_FREESLIP:
                        bc->periodic |= SPACE_FREESLIP_X;
                        break;
                }
                
                switch(init_bc(&(bc->front), args)) {
                    case BOUNDARY_PERIODIC:
                        bc->periodic |= space_periodic_y;
                        break;
                    case BOUNDARY_FREESLIP:
                        bc->periodic |= SPACE_FREESLIP_Y;
                        break;
                }
                
                switch(init_bc(&(bc->back), args)) {
                    case BOUNDARY_PERIODIC:
                        bc->periodic |= space_periodic_y;
                        break;
                    case BOUNDARY_FREESLIP:
                        bc->periodic |= SPACE_FREESLIP_Y;
                        break;
                }
                
                switch(init_bc(&(bc->top), args)) {
                    case BOUNDARY_PERIODIC:
                        bc->periodic |= space_periodic_z;
                        break;
                    case BOUNDARY_FREESLIP:
                        bc->periodic |= SPACE_FREESLIP_Z;
                        break;
                }
                
                switch(init_bc(&(bc->bottom), args)) {
                    case BOUNDARY_PERIODIC:
                        bc->periodic |= space_periodic_z;
                        break;
                    case BOUNDARY_FREESLIP:
                        bc->periodic |= SPACE_FREESLIP_Z;
                        break;
                }
                
                check_periodicy(&(bc->left), &(bc->right));
                check_periodicy(&(bc->front), &(bc->back));
                check_periodicy(&(bc->top), &(bc->bottom));
            }
            else {
                std::string msg = "invalid type, (";
                msg += args->ob_type->tp_name;
                msg += ") given for boundary condition init, only integer, string or dictionary allowed";
                throw std::logic_error(msg);
            }
        }
        catch(const std::exception &e) {
            return C_EXP(e);
        }
    }
    // no args given, use default init value
    else {
        // default value
        bc->periodic = space_periodic_full;
        boundaries_from_flags(bc);
    }
    
    if(cells[0] < 3 && (bc->periodic & space_periodic_x)) {
        cells[0] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[0]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        PyErr_WarnEx(NULL, msg.c_str(), 0);
    }
    if(cells[1] < 3 && (bc->periodic & space_periodic_y)) {
        cells[1] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[1]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        PyErr_WarnEx(NULL, msg.c_str(), 0);
    }
    if(cells[2] < 3 && (bc->periodic & space_periodic_z)) {
        cells[2] = 3;
        std::string msg = "requested periodic_x and " + std::to_string(cells[2]) +
        " space cells in the x direction, need at least 3 cells for periodic, setting cell count to 3";
        PyErr_WarnEx(NULL, msg.c_str(), 0);
    }
    
    printf("engine periodic x : %s\n", bc->periodic & space_periodic_x ? "true" : "false");
    printf("engine periodic y : %s\n", bc->periodic & space_periodic_y ? "true" : "false");
    printf("engine periodic z : %s\n", bc->periodic & space_periodic_z ? "true" : "false");
    printf("engine freeslip x : %s\n", bc->periodic & SPACE_FREESLIP_X ? "true" : "false");
    printf("engine freeslip y : %s\n", bc->periodic & SPACE_FREESLIP_Y ? "true" : "false");
    printf("engine freeslip z : %s\n", bc->periodic & SPACE_FREESLIP_Z ? "true" : "false");
    printf("engine periodic ghost x : %s\n", bc->periodic & space_periodic_ghost_x ? "true" : "false");
    printf("engine periodic ghost y : %s\n", bc->periodic & space_periodic_ghost_y ? "true" : "false");
    printf("engine periodic ghost z : %s\n", bc->periodic & space_periodic_ghost_z ? "true" : "false");
    
    return S_OK;
        

}

int MxBoundaryCondition_Check(const PyObject *obj)
{
    return obj && obj->ob_type == &MxBoundaryCondition_Type;
}

int MxBoundaryConditions_Check(const PyObject *obj)
{
    return obj && obj->ob_type == &MxBoundaryConditions_Type;
}

HRESULT _MxBoundaryConditions_Init(PyObject* m) {
    if (PyType_Ready((PyTypeObject*)&MxBoundaryCondition_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&MxBoundaryCondition_Type);
    if (PyModule_AddObject(m, "BoundaryCondition", (PyObject *)&MxBoundaryCondition_Type) < 0) {
        Py_DECREF(&MxBoundaryCondition_Type);
        return E_FAIL;
    }

    if (PyType_Ready((PyTypeObject*)&MxBoundaryConditions_Type) < 0) {
        return E_FAIL;
    }

    Py_INCREF(&MxBoundaryConditions_Type);
    if (PyModule_AddObject(m, "BoundaryConditions", (PyObject *)&MxBoundaryConditions_Type) < 0) {
        Py_DECREF(&MxBoundaryConditions_Type);
        return E_FAIL;
    }

    return S_OK;
}


PyObject* bc_str(MxBoundaryCondition *bc) {
    return mx::cast(bc->str(true));
}

PyObject* bcs_str(MxBoundaryConditions *bcs) {
    std::string s = "BoundaryConditions(\n";
    s += "  " + bcs->left.str(false) + ", \n";
    s += "  " + bcs->right.str(false) + ", \n";
    s += "  " + bcs->front.str(false) + ", \n";
    s += "  " + bcs->back.str(false) + ", \n";
    s += "  " + bcs->bottom.str(false) + ", \n";
    s += "  " + bcs->top.str(false) + ", \n";
    s += ")";
    return mx::cast(s);
}

void MxBoundaryCondition::set_potential(struct MxParticleType *ptype,
        struct MxPotential *pot)
{
    int i = ptype->id;
    if(potenntials[i]) {
        Py_DECREF(potenntials[i]);
    }
    
    potenntials[i] = pot;
    
    if(pot) {
        Py_INCREF(pot);
    }
}

std::string MxBoundaryCondition::str(bool show_type) const
{
    std::string s;
    
    if(show_type) {
        s +=  "BoundaryCondition(";
    }
    
    s += "\'";
    s += this->name;
    s += "\' : {";
    
    s += "\'kind\' : \'";
    switch (kind) {
        case BOUNDARY_PERIODIC:
            s += "PERIODIC";
            break;
        case BOUNDARY_FORCE:
            s += "FORCE";
            break;
        case BOUNDARY_FREESLIP:
            s += "FREESLIP";
            break;
        case BOUNDARY_VELOCITY:
            s += "VELOCITY";
            break;
        default:
            s += "INVALID";
            break;
    }
    s += "\'";
    s += ", \'velocity\' : [" + std::to_string(velocity[0]) + ", " + std::to_string(velocity[1]) + ", " + std::to_string(velocity[2]) + "]";
    s += ", \'restore\' : " + std::to_string(restore);
    s += "}";
    
    if(show_type) {
        s +=  ")";
    }
    
    return s;
}

void MxBoundaryConditions::set_potential(struct MxParticleType *ptype,
        struct MxPotential *pot)
{
    left.set_potential(ptype, pot);
    right.set_potential(ptype, pot);
    front.set_potential(ptype, pot);
    back.set_potential(ptype, pot);
    bottom.set_potential(ptype, pot);
    top.set_potential(ptype, pot);
}
