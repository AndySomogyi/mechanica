/*
 * MxColorMapper.cpp
 *
 *  Created on: Dec 27, 2020
 *      Author: andy
 */

#include <rendering/MxColorMapper.hpp>
#include "MxParticle.h"
#include <CSpecies.hpp>
#include <CSpeciesList.hpp>
#include <CStateVector.hpp>
#include <CConvert.hpp>

#include "colormaps/colormaps.h"

static Magnum::Color4 bgyr_35_85_c72(MxColorMapper *cm, struct MxParticle *part) {
    float s = part->state_vector->fvec[0];
    return Magnum::Color4{colormaps::all::bgyr_35_85_c72(s), 1};
}

MxColorMapper *MxColorMapper_New(struct MxParticleType *partType,
                                 const char* speciesName,
                                 const char* name, float min, float max) {
    MxColorMapper *obj = new MxColorMapper();
    
    obj->map = bgyr_35_85_c72;
    
    return obj;
}

MxColorMapper *MxColorMapper_New(PyObject *args, PyObject *kwargs) {
    if(args == nullptr) {
        PyErr_WarnEx(PyExc_Warning, "args to MxColorMapper_New is NULL", 2);
        return NULL;
    }
    
    MxParticleType *type = MxParticleType_Get(args);
    if(type == nullptr) {
        PyErr_WarnEx(PyExc_Warning, "args to MxColorMapper_New is not a ParticleType", 2);
        return NULL;
    }
    
    if(type->species == NULL) {
        PyErr_WarnEx(PyExc_Warning, "can't create color map on a type without any species", 2);
        return NULL;
    }
    
    MxColorMapper *obj = NULL;
    
    
    
    try {
        // always needs a species
        std::string species = carbon::cast<std::string>(PyDict_GetItemString(kwargs, "species"));
        
        PyObject *pmap = PyDict_GetItemString(kwargs, "map");
        
        std::string map = pmap ? carbon::cast<std::string>(pmap) : "rainbow";
        
        return MxColorMapper_New(type,
                          species.c_str(),
                                 map.c_str(),
                                 0, 1);
        
        
    }
    catch(const std::exception &ex) {
        delete obj;
        PyErr_WarnEx(PyExc_Warning, ex.what(), 2);
        return NULL;
    }
    
    return obj;
}


