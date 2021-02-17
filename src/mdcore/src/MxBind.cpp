/*
 * MxPotentialBind.cpp
 *
 *  Created on: Feb 13, 2021
 *      Author: andy
 */

#include <MxBind.hpp>
#include <MxParticle.h>
#include <engine.h>
#include <bond.h>
#include <string>

static PyObject *potential_bind_ptype_ptype();

static PyObject *potential_bind_ptype_boundary(MxPotential *pot, MxParticleType *ptype, MxBoundaryCondition *bc);

HRESULT universe_bind_potential(MxPotential *p, PyObject *a, PyObject *b, bool bound) {
    MxParticleType *a_type = MxParticleType_Get(a);
    MxParticleType *b_type = MxParticleType_Get(b);
    if(a_type && b_type) {

        MxPotential *pot = NULL;

        if(p->create_func) {
            pot = p->create_func(p, (MxParticleType*)a, (MxParticleType*)b);
        }
        else {
            pot = p;
        }

        if(bound) {
            pot->flags = pot->flags | POTENTIAL_BOUND;
        }

        if(engine_addpot(&_Engine, pot, a_type->id, b_type->id) != engine_err_ok) {
            std::string msg = "failed to add potential to engine: error";
            msg += std::to_string(engine_err);
            msg += ", ";
            msg += engine_err_msg[-engine_err];
            return mx_error(E_FAIL, msg.c_str());
        }
        return S_OK;
    }

    if(MxParticle_Check(a) && MxParticle_Check(b)) {
        MxParticleHandle *a_part = ((MxParticleHandle *)a);
        MxParticleHandle *b_part = ((MxParticleHandle *)b);

        //MxBond_New(uint32_t flags,
        //        int32_t i, int32_t j,
        //        double half_life,
        //        double bond_energy,
        //        struct MxPotential* potential);

        MxBondHandle_New(0, a_part->id, b_part->id,
                std::numeric_limits<double>::max(),
                std::numeric_limits<double>::max(),
                p);

        return S_OK;
    }

    if(MxCuboidType_Check(a) && b_type) {
        return engine_add_cuboid_potential(&_Engine, p, b_type->id);
    }

    if(MxCuboidType_Check(b) && a_type) {
        return engine_add_cuboid_potential(&_Engine, p, a_type->id);
    }
    
    if(MxBoundaryConditions_Check(a) && b_type) {
        MxBoundaryConditions *bc = (MxBoundaryConditions*)a;
        bc->set_potential(b_type, p);
        return S_OK;
    }
    
    if(MxBoundaryConditions_Check(b) && a_type) {
        MxBoundaryConditions *bc = (MxBoundaryConditions*)b;
        bc->set_potential(a_type, p);
        return S_OK;
    }
    
    if(MxBoundaryCondition_Check(a) && b_type) {
        MxBoundaryCondition *bc = (MxBoundaryCondition*)a;
        bc->set_potential(b_type, p);
        return S_OK;
    }
    
    if(MxBoundaryCondition_Check(b) && a_type) {
        MxBoundaryCondition *bc = (MxBoundaryCondition*)b;
        bc->set_potential(a_type, p);
        return S_OK;
    }
    
    return mx_error(E_FAIL, "can only add potential to particle types or instances");
}

PyObject *potential_bind_ptype_boundary(MxPotential *pot, MxParticleType *ptype, MxBoundaryCondition *bc) {
    
}


PyObject *MxPotential_Bind(MxPotential *pot, PyObject *args, PyObject *kwargs) {
    return NULL;
}
