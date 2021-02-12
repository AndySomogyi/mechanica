/*
 * BoundaryConditions.cpp
 *
 *  Created on: Feb 10, 2021
 *      Author: andy
 */

#include "MxBoundaryConditions.hpp"
#include "MxConvert.hpp"
#include "space.h"


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
    .tp_getset =         0,
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
    .tp_getset =         0,
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

void test(MxBoundaryConditions *bc) {
    bc->periodic = space_periodic_x | SPACE_FREESLIP_Y;
    
    bc->left.kind = BOUNDARY_PERIODIC;
    bc->right.kind = BOUNDARY_PERIODIC;
    
    bc->front.kind = BOUNDARY_FREESLIP;
    bc->back.kind = BOUNDARY_FREESLIP;
    
    bc->top.kind = BOUNDARY_VELOCITY;
    bc->bottom.kind = BOUNDARY_VELOCITY;
    
}


HRESULT MxBoundaryConditions_Init(MxBoundaryConditions *bc, int *cells, PyObject *args) {
    
    try {
        
        if(args && mx::check<int>(args)) {
            bc->periodic = mx::cast<uint32_t>(args);
            boundaries_from_flags(bc);
        }
        else if(args && PyDict_Check(args)) {
            
        }
        else {
            // default value
            bc->periodic = space_periodic_full;
            boundaries_from_flags(bc);
        }
        
        test(bc);
        
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
    catch(const std::exception &e) {
        return C_EXP(e);
    }
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


