/*
 * MxParticleEvent.cpp
 *
 *  Created on: Jun 25, 2020
 *      Author: andy
 */

#include <MxParticleEvent.h>
#include <MxParticle.h>
#include <engine.h>
#include <space.h>
#include <iostream>
#include <CConvert.hpp>
#include <MxPy.h>

static HRESULT particletimeevent_pyfunction_invoke_uniform_random(CTimeEvent *event, double time);

static HRESULT particletimeevent_pyfunction_invoke_largest(CTimeEvent *event, double time);

static HRESULT particletimeevent_pyfunction_invoke_oldest(CTimeEvent *event, double time);

static HRESULT particletimeevent_exponential_setnexttime(CTimeEvent *event, double time);

static HRESULT particletimeevent_fixed_setnexttime(CTimeEvent *event, double time);


MxParticleType *MxParticleType_FromMethod(PyObject *e);

PyObject* MxOnTime(PyObject *module, PyObject *args, PyObject *kwargs)
{
    Log(LOG_TRACE) << "obj: " << PyUnicode_AsUTF8AndSize(PyObject_Str(module), NULL) << std::endl
                   << "args: " << PyUnicode_AsUTF8AndSize(PyObject_Str(args), NULL) << std::endl
                   << "kwargs: " << PyUnicode_AsUTF8AndSize(PyObject_Str(kwargs), NULL);
    
    MxParticleTimeEvent* event = (MxParticleTimeEvent*)CTimeEvent_New();
    
    if(CTimeEvent_Init(event, args, kwargs) != 0) {
        Py_DECREF(event);
        return NULL;
    }
    
    CMulticastTimeEvent_Add(_Engine.on_time, event);
    
    MxParticleType *partType = MxParticleType_FromFunction(event->method);
    
    if(partType) {
        MxParticleType_BindEvent(partType, event);
    }
    
    return event;
}


/**
 * static method to create an on_time event object
 */



MxParticleTimeEvent *MxParticleTimeEvent_New(PyObject *args, PyObject *kwargs) {
    throw std::logic_error("not impleented");
    return NULL;
}

HRESULT _MxTimeEvent_Init(PyObject *m)
{
    return  S_OK;
}

HRESULT MxParticleTimeEvent_BindParticleMethod(CTimeEvent *event,
                                               struct MxParticleType *target, PyObject *method)
{
    
    Log(LOG_TRACE) << "target: " << PyUnicode_AsUTF8AndSize(PyObject_Str((PyObject*)target), NULL) << std::endl
                   << "method: " << PyUnicode_AsUTF8AndSize(PyObject_Str(method), NULL);
    
    
    return E_NOTIMPL;
}

static std::vector<std::string> split(const std::string& s, char seperator)
{
    std::vector<std::string> output;
    
    std::string::size_type prev_pos = 0, pos = 0;
    
    while((pos = s.find(seperator, pos)) != std::string::npos)
    {
        std::string substring( s.substr(prev_pos, pos-prev_pos) );
        
        output.push_back(substring);
        
        prev_pos = ++pos;
    }
    
    output.push_back(s.substr(prev_pos, pos-prev_pos)); // Last word
    
    return output;
}


MxParticleType *MxParticleType_FromFunction(PyObject *e) {
    
    if(PyFunction_Check(e)) {
        
        PyFunctionObject *func = (PyFunctionObject*)e;
        
        if(func->func_module == NULL ||
           func->func_qualname == NULL ||
           func->func_name == NULL) {
            return NULL;
        }
        
        try {
            std::string name = carbon::cast<std::string>(func->func_name);
            std::string qualname = carbon::cast<std::string>(func->func_qualname);
            
            PyObject *module = PyImport_GetModule(func->func_module);
            
            std::vector<std::string> names = split(qualname, '.');
            
            if(names.size() > 1) {
                
                PyObject *module_dict = PyModule_GetDict(module);
                
                std::string ownerName = names[names.size() - 2];
                
                PyObject *owner = PyDict_GetItemString(module_dict, ownerName.c_str());
                
                // if the owner name is not empty, but looking up the name in the function
                // module is null, that means that on_time was called inside
                // a class defition.
                // TODO: don't support this yet, but can
                // add a decorator and process them when the type gets
                // created.
                
                if(owner &&
                   PyType_Check(owner) &&
                   PyObject_IsSubclass(owner, (PyObject*)MxParticle_GetType()) > 0) {
                    return (MxParticleType*)owner;
                }
            }
        }
        catch(const std::exception &e) {
        }
        
        return NULL;
    }
    else if (Py_TYPE(e) == &PyMethodDescr_Type) {
        
        PyDescrObject *d = (PyDescrObject*)e;
        
        if(PyType_Check(d->d_type) &&
           PyObject_IsSubclass((PyObject*)d->d_type, (PyObject*)MxParticle_GetType()) > 0) {
            return (MxParticleType*)d->d_type;
        }
        
        return NULL;
    }
    return NULL;
}



HRESULT MxParticleType_BindEvent(MxParticleType *type, PyObject *e) {
    
    if(PySequence_Check(e)) {
        // TODO: what here???
    }
    
    else if(PyObject_IsInstance(e, (PyObject*)&CTimeEvent_Type)) {
        
        CTimeEvent *timeEvent = (CTimeEvent*)e;
        
        if(timeEvent->target) {
            return c_error(E_FAIL, "event target already set in particle type definition");
        }
        
        timeEvent->target = (PyObject*)type;
        Py_INCREF(timeEvent->target);
        
        timeEvent->flags |= EVENT_ACTIVE;
        
        if(timeEvent->predicate && PyUnicode_Check(timeEvent->predicate)) {
            
            if(PyUnicode_CompareWithASCIIString(timeEvent->predicate, "largest") == 0) {
                timeEvent->te_invoke = (timeevent_invoke)particletimeevent_pyfunction_invoke_largest;
            }
            else {
                return mx_error(E_FAIL, "invalid predicate option");
            }
        }
        else {
            timeEvent->te_invoke = (timeevent_invoke)particletimeevent_pyfunction_invoke_uniform_random;
        }
        
        if(timeEvent->flags & EVENT_EXPONENTIAL) {
            timeEvent->te_setnexttime = particletimeevent_exponential_setnexttime;
            timeEvent->te_setnexttime(timeEvent, _Engine.time * _Engine.dt);
        }
        else {
            timeEvent->te_setnexttime = particletimeevent_fixed_setnexttime;
            timeEvent->te_setnexttime(timeEvent, _Engine.time * _Engine.dt);
        }
    }
    
    return S_OK;
}

HRESULT MyParticleType_BindEvents(struct MxParticleType *type, PyObject *events)
{
    Log(LOG_DEBUG) << "type: " << PyUnicode_AsUTF8AndSize(PyObject_Str((PyObject*)type), NULL) << std::endl
                   << "events: " << PyUnicode_AsUTF8AndSize(PyObject_Str(events), NULL) << std::endl;
    
    if (PySequence_Check(events) == 0) {
        return c_error(E_FAIL, "events must be a list");
    }
    
    for(int i = 0; i < PySequence_Size(events); ++i) {
        PyObject *e = PySequence_Fast_GET_ITEM(events, i);
        
        HRESULT r = MxParticleType_BindEvent(type, e);
        
        if(!SUCCEEDED(r)) {
            return r;
        }
    }
    return S_OK;
}

HRESULT particletimeevent_pyfunction_invoke_largest(CTimeEvent *event, double time) {
    
    MxParticleType *type = (MxParticleType*)event->target;
    
    if(type->parts.nr_parts == 0) {
        return S_OK;
    }
    
    // TODO: memory leak
    PyObject *args = PyTuple_New(2);
    
    // max particle
    MxParticle *mp = type->particle(0);
    // find the object with the largest number of contained objects
    for(int i = 1; i < type->parts.nr_parts; ++i) {
        MxParticle *part = type->particle(i);
        if(part->nr_parts > mp->nr_parts) {
            mp = part;
        }
    }
    
    PyObject *t = PyFloat_FromDouble(time);
    PyTuple_SET_ITEM(args, 0, mp->_pyparticle);
    PyTuple_SET_ITEM(args, 1, t);
    
    Log(LOG_TRACE) << "args: " << PyUnicode_AsUTF8AndSize(PyObject_Str(args), NULL) << std::endl
                   << "method: " << PyUnicode_AsUTF8AndSize(PyObject_Str(event->method), NULL);
    
    // time expired, so invoke the event.
    PyObject *result = PyObject_CallObject((PyObject*)event->method, args);
    
    Py_DecRef(result);
    
    return S_OK;
}


HRESULT particletimeevent_pyfunction_invoke_uniform_random(CTimeEvent *event, double time) {
    
    MxParticleType *type = (MxParticleType*)event->target;
    
    if(type->parts.nr_parts == 0) {
        return S_OK;
    }
    
    std::uniform_int_distribution<int> distribution(0,type->parts.nr_parts-1);
    
    // TODO: memory leak
    PyObject *args = PyTuple_New(2);
    
    // index in the type's list of particles
    int tid = distribution(CRandom);
    
    int pid = type->parts.parts[tid];
    
    assert(_Engine.s.partlist[pid]);
    assert(_Engine.s.partlist[pid]->_pyparticle);
    
    PyObject *t = PyFloat_FromDouble(time);
    PyTuple_SET_ITEM(args, 0, _Engine.s.partlist[pid]->_pyparticle);
    PyTuple_SET_ITEM(args, 1, t);
    
    // time expired, so invoke the event.
    // TODO: major memory leak, check result
    PyObject *result = PyObject_CallObject((PyObject*)event->method, args);
    
    return S_OK;
}

PyObject *MxInvokeTime(PyObject *module, PyObject *args, PyObject *kwargs) {
    
    double time = PyFloat_AsDouble(PyTuple_GetItem(args, 0));
    
    // TODO: check return
    CMulticastTimeEvent_Invoke(_Engine.on_time, time);
    
    Py_RETURN_NONE;
}

// need to scale period by number of particles.
// TODO: need to update next time when particles are added or removed.
HRESULT particletimeevent_exponential_setnexttime(CTimeEvent *event, double time) {
    double rescale = 1;
    if(event->flags & EVENT_PERIOD_RESCALE) {
        MxParticleType *type = (MxParticleType*)event->target;
        rescale = type->parts.nr_parts > 0 ? type->parts.nr_parts : 1;
    }
    std::exponential_distribution<> d(rescale / event->period);
    event->next_time = time + d(CRandom);
    return S_OK;
}


// need to scale period by number of particles.
// TODO: need to update next time when particles are added or removed.
HRESULT particletimeevent_fixed_setnexttime(CTimeEvent *event, double time) {
    if(event->flags & EVENT_PERIOD_RESCALE) {
        MxParticleType *type = (MxParticleType*)event->target;
        uint32_t nr_parts = type->parts.nr_parts > 0 ? type->parts.nr_parts : 1;
        event->next_time = time + (event->period / nr_parts);
    }
    else {
        event->next_time = time + event->period;
    }
    return S_OK;
}
