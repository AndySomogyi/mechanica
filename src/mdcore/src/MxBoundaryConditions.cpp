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
        .name = "normal",
        .get = [](PyObject *obj, void *p) -> PyObject* {
            MxBoundaryCondition *bc = (MxBoundaryCondition*)obj;
            return mx::cast(bc->normal);
        },
        .set = [](PyObject *obj, PyObject *val, void *p) -> int {
            PyErr_SetString(PyExc_TypeError, "readonly property");
            return -1;
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
        
        else if (s.compare("POTENTIAL") == 0 ) {
            result = BOUNDARY_POTENTIAL | BOUNDARY_FREESLIP;
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

static unsigned init_bc_direction(MxBoundaryCondition *low_bl, MxBoundaryCondition *high_bl, PyObject *o) {
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
    
    int s = 6 * engine::max_type * sizeof(MxPotential**);
    
    bc->potenntials = (MxPotential**)malloc(6 * engine::max_type * sizeof(MxPotential**));
    bzero(bc->potenntials, 6 * engine::max_type * sizeof(MxPotential**));
    
    bc->left.name = "left";     bc->left.restore = 1.f;     bc->left.potenntials =   &bc->potenntials[0 * engine::max_type];
    bc->right.name = "right";   bc->right.restore = 1.f;    bc->right.potenntials =  &bc->potenntials[1 * engine::max_type];
    bc->front.name = "front";   bc->front.restore = 1.f;    bc->front.potenntials =  &bc->potenntials[2 * engine::max_type];
    bc->back.name = "back";     bc->back.restore = 1.f;     bc->back.potenntials =   &bc->potenntials[3 * engine::max_type];
    bc->top.name = "top";       bc->top.restore = 1.f;      bc->top.potenntials =    &bc->potenntials[4 * engine::max_type];
    bc->bottom.name = "bottom"; bc->bottom.restore = 1.f;   bc->bottom.potenntials = &bc->potenntials[5 * engine::max_type];
    
    bc->left.normal =   { 1.f,  0.f,  0.f};
    bc->right.normal =  {-1.f,  0.f,  0.f};
    bc->front.normal =  { 0.f,  1.f,  0.f};
    bc->back.normal =   { 0.f, -1.f,  0.f};
    bc->bottom.normal = { 0.f,  0.f,  1.f};
    bc->top.normal =    { 0.f,  0.f, -1.f};

    if(args) {
        try {
            if(mx::check<int>(args)) {
                int value = mx::cast<int>(args);
                
                int test = SPACE_FREESLIP_FULL;
                
                
                switch(value) {
                    case space_periodic_none :
                    case space_periodic_x:
                    case space_periodic_y:
                    case space_periodic_z:
                    case space_periodic_full:
                    case space_periodic_ghost_x:
                    case space_periodic_ghost_y:
                    case space_periodic_ghost_z:
                    case space_periodic_ghost_full:
                    case SPACE_FREESLIP_X:
                    case SPACE_FREESLIP_Y:
                    case SPACE_FREESLIP_Z:
                    case SPACE_FREESLIP_FULL:
                        bc->periodic = value;
                        break;
                    default: {
                        std::string msg = "invalid value " + std::to_string(value) + ", for integer boundary condition";
                        throw std::logic_error(msg);
                    }
                }
                
                boundaries_from_flags(bc);
            }
            else if(PyDict_Check(args)) {
                // check if we have x, y, z directions
                PyObject *o;
                unsigned dir;

                if((o = PyDict_GetItemString(args, "x"))) {
                    dir = init_bc_direction(&(bc->left), &(bc->right), o);
                    if(dir & BOUNDARY_PERIODIC) {
                        bc->periodic |= space_periodic_x;
                    }
                    if(dir & BOUNDARY_FREESLIP) {
                        bc->periodic |= SPACE_FREESLIP_X;
                    }
                }

                if((o = PyDict_GetItemString(args, "y"))) {
                    dir = init_bc_direction(&(bc->front), &(bc->back), o);
                    if(dir & BOUNDARY_PERIODIC) {
                        bc->periodic |= space_periodic_y;
                    }
                    if(dir & BOUNDARY_FREESLIP) {
                        bc->periodic |= SPACE_FREESLIP_Y;
                    }
                }

                if((o = PyDict_GetItemString(args, "z"))) {
                    dir = init_bc_direction(&(bc->top), &(bc->bottom), o);
                    if(dir & BOUNDARY_PERIODIC) {
                        bc->periodic |= space_periodic_z;
                    }
                    if(dir & BOUNDARY_FREESLIP) {
                        bc->periodic |= SPACE_FREESLIP_Z;
                    }
                }

                dir = init_bc(&(bc->left), args);
                if(dir & BOUNDARY_PERIODIC) {
                    bc->periodic |= space_periodic_x;
                }
                if(dir & BOUNDARY_FREESLIP) {
                    bc->periodic |= SPACE_FREESLIP_X;
                }

                dir = init_bc(&(bc->right), args);
                if(dir & BOUNDARY_PERIODIC) {
                    bc->periodic |= space_periodic_x;
                }
                if(dir & BOUNDARY_FREESLIP) {
                    bc->periodic |= SPACE_FREESLIP_X;
                }

                dir = init_bc(&(bc->front), args);
                if(dir & BOUNDARY_PERIODIC) {
                    bc->periodic |= space_periodic_y;
                }
                if(dir & BOUNDARY_FREESLIP) {
                    bc->periodic |= SPACE_FREESLIP_Y;
                }

                dir = init_bc(&(bc->back), args);
                if(dir & BOUNDARY_PERIODIC) {
                    bc->periodic |= space_periodic_y;
                }
                if(dir & BOUNDARY_FREESLIP) {
                    bc->periodic |= SPACE_FREESLIP_Y;
                }

                dir = init_bc(&(bc->top), args);
                if(dir & BOUNDARY_PERIODIC) {
                    bc->periodic |= space_periodic_z;
                }
                if(dir & BOUNDARY_FREESLIP) {
                    bc->periodic |= SPACE_FREESLIP_Z;
                }

                dir = init_bc(&(bc->bottom), args);
                if(dir & BOUNDARY_PERIODIC) {
                    bc->periodic |= space_periodic_z;
                }
                if(dir & BOUNDARY_FREESLIP) {
                    bc->periodic |= SPACE_FREESLIP_Z;
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
    
    Log(LOG_INFORMATION) << "engine periodic x : " << (bool)(bc->periodic & space_periodic_x) ;
    Log(LOG_INFORMATION) << "engine periodic y : " << (bool)(bc->periodic & space_periodic_y) ;
    Log(LOG_INFORMATION) << "engine periodic z : " << (bool)(bc->periodic & space_periodic_z) ;
    Log(LOG_INFORMATION) << "engine freeslip x : " << (bool)(bc->periodic & SPACE_FREESLIP_X) ;
    Log(LOG_INFORMATION) << "engine freeslip y : " << (bool)(bc->periodic & SPACE_FREESLIP_Y) ;
    Log(LOG_INFORMATION) << "engine freeslip z : " << (bool)(bc->periodic & SPACE_FREESLIP_Z) ;
    Log(LOG_INFORMATION) << "engine periodic ghost x : " << (bool)(bc->periodic & space_periodic_ghost_x) ;
    Log(LOG_INFORMATION) << "engine periodic ghost y : " << (bool)(bc->periodic & space_periodic_ghost_y) ;
    Log(LOG_INFORMATION) << "engine periodic ghost z : " << (bool)(bc->periodic & space_periodic_ghost_z) ;
    
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
    
    PyModule_AddIntConstant(m, "BOUNDARY_NONE",       space_periodic_none);
    PyModule_AddIntConstant(m, "PERIODIC_X",          space_periodic_x);
    PyModule_AddIntConstant(m, "PERIODIC_Y",          space_periodic_y);
    PyModule_AddIntConstant(m, "PERIODIC_Z",          space_periodic_z);
    PyModule_AddIntConstant(m, "PERIODIC_FULL",       space_periodic_full);
    PyModule_AddIntConstant(m, "PERIODIC_GHOST_X",    space_periodic_ghost_x);
    PyModule_AddIntConstant(m, "PERIODIC_GHOST_Y",    space_periodic_ghost_y);
    PyModule_AddIntConstant(m, "PERIODIC_GHOST_Z",    space_periodic_ghost_z);
    PyModule_AddIntConstant(m, "PERIODIC_GHOST_FULL", space_periodic_ghost_full);
    PyModule_AddIntConstant(m, "FREESLIP_X",          SPACE_FREESLIP_X);
    PyModule_AddIntConstant(m, "FREESLIP_Y",          SPACE_FREESLIP_Y);
    PyModule_AddIntConstant(m, "FREESLIP_Z",          SPACE_FREESLIP_Z);
    PyModule_AddIntConstant(m, "FREESLIP_FULL",       SPACE_FREESLIP_FULL);
    
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
        case BOUNDARY_POTENTIAL:
            s += "POTENTIAL";
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
