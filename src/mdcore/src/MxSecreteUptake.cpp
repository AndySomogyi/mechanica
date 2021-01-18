/*
 * MxSecreteUptake.cpp
 *
 *  Created on: Jan 6, 2021
 *      Author: andy
 */

#include <MxSecreteUptake.hpp>
#include <MxParticle.h>
#include <CSpeciesValue.hpp>
#include <CSpeciesList.hpp>
#include <CStateVector.hpp>
#include <CConvert.hpp>
#include <MxParticleList.hpp>
#include <metrics.h>
#include <engine.h>
#include <iostream>





HRESULT MxSecrete_AmountToParticles(struct CSpeciesValue *species,
        float amount, uint16_t nr_parts,
        int32_t *parts, float *secreted)
{
    CStateVector *stateVector = species->state_vector;
    CSpecies *s = stateVector->species->item(species->index);
    const std::string& speciesName = s->getId();
    
    float amountToRemove = amount < stateVector->fvec[species->index] ? amount : stateVector->fvec[species->index];
    
    struct ParticleId {
        MxParticle *part;
        int32_t index;
    };
    
    std::vector<ParticleId> pvec;
    
    for(int i = 0; i < nr_parts; ++i) {
        MxParticle *p = _Engine.s.partlist[parts[i]];
        
        int index;
        
        if(p && p->state_vector && (index = p->state_vector->species->index_of(speciesName)) >= 0) {
            pvec.push_back({p, index});
        }
    }
    
    if(pvec.size() > 0) {
        float amountPer = amountToRemove / pvec.size();
        for(ParticleId& p : pvec) {
            p.part->state_vector->fvec[p.index] += amountPer;
        }
        stateVector->fvec[species->index] -= amountToRemove;
        if(secreted) {
            *secreted = amountToRemove;
        }
    }
    
    return S_OK;
}

HRESULT MxSecrete_AmountWithinDistance(struct CSpeciesValue *species,
        float amount, float radius,
        const std::set<short int> *typeIds, float *secreted)
{
    MxParticle *part = MxParticle_Get(species->state_vector->owner);
    uint16_t nr_parts = 0;
    int32_t *parts = NULL;
    
    MxParticle_Neighbors(part, radius, typeIds, &nr_parts, &parts);
    
    return MxSecrete_AmountToParticles(species, amount, nr_parts, parts, secreted);
}



static PyObject *secrete(PyObject *self, PyObject *args, PyObject *kwargs) {
    
    float secreted = 0;
    PyObject *result = NULL;
    CSpeciesValue *species = (CSpeciesValue*)self;
    MxParticle *part = MxParticle_Get(species->state_vector->owner);
    if(!part) {
        PyErr_SetString(PyExc_SystemError, "species state vector has no owner");
        return NULL;
    }
    
    try{
        //std::cout << "secrete(self: "
        //    << carbon::repr(self)
        //    << ", args:"
        //    << carbon::repr(args)
        //    << ", kwargs: "
        //    << carbon::repr(kwargs)
        //    << ")"
        //    << std::endl;
        
        float amount = carbon::cast<float>(carbon::py_arg("amount", 0, args, kwargs));
        
        PyObject *to = carbon::py_arg("to", 1, args, kwargs);
        
        MxParticleList *toList = MxParticleList_FromPyObject(to);
        
        if(toList) {
            if(SUCCEEDED(MxSecrete_AmountToParticles(species, amount, toList->nr_parts,
                                                     toList->parts, &secreted))) {
                result = carbon::cast(secreted);
            }
            Py_DECREF(toList);
            return result;
        }
        
        PyObject *distance = carbon::py_arg("distance", 1, args, kwargs);
        
        if(carbon::check<float>(distance)) {
            
            // take into account the radius of this particle.
            float radius = part->radius + carbon::cast<float>(distance);
            
            std::set<short int> ids;
            
            if(FAILED(MxParticleType_IdsFromPythonObj(NULL, ids))) {
                PyErr_SetString(PyExc_SystemError, "error getting particle ids");
                return NULL;
            }
            
            if(FAILED(MxSecrete_AmountWithinDistance(species, amount, radius, &ids, &secreted))) {
                return NULL;
            }
            
            result = carbon::cast(secreted);
        }
    }
    catch(const std::exception &e) {
        
    }
    
    return result;
}

PyMethodDef secrete_mthdef = {
    "secrete",                          // const char  *ml_name;   /* The name of the built-in function/method */
    (PyCFunction)secrete,               // PyCFunction ml_meth;    /* The C function that implements it */
    METH_VARARGS | METH_KEYWORDS,       // ml_flags;   /* Combination of METH_xxx flags, which mostly
                                        //                describe the args expected by the C func */
    "docs..."                           //const char  *ml_doc;     /* The __doc__ attribute, or NULL */
};


HRESULT _MxSecreteUptake_Init(PyObject *m)
{
    if(PyType_Ready(&CSpeciesValue_Type) != 0) {
        return c_error(E_FAIL, "CSpeciesValue_Type is not ready");
    }
    
    PyObject *descr = PyDescr_NewMethod(&CSpeciesValue_Type, &secrete_mthdef);
    if(!descr) {
        return c_error(E_FAIL, "could not create secrete method");
    }
    
    if(PyDict_SetItemString(CSpeciesValue_Type.tp_dict, secrete_mthdef.ml_name, descr) != 0) {
        return c_error(E_FAIL, "error setting CSpeciesValue_Type dictionary secrete value");
    }
    
    Py_DECREF(descr);
    
    return S_OK;
}
