/*
 * MxParticleEvent.h
 *
 *  Created on: Jun 25, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_MXPARTICLEEVENT_H_
#define SRC_MDCORE_INCLUDE_MXPARTICLEEVENT_H_

#include <CEvent.hpp>


struct MxParticleTimeEvent : CTimeEvent {
};


MxParticleTimeEvent *MxParticleTimeEvent_New(PyObject *args, PyObject *kwargs);


HRESULT MyParticleType_BindEvents(struct MxParticleType *type, PyObject *events);


HRESULT MxParticleTimeEvent_BindParticleMethod(CTimeEvent *event,
        struct MxParticleType *target, PyObject *method);


PyObject *MxOnTime(PyObject *module, PyObject *obj, PyObject *args);

PyObject *MxInvokeTime(PyObject *module, PyObject *obj, PyObject *args);

HRESULT MxParticleType_BindEvent(struct MxParticleType *type, PyObject *e);

/**
 * Check if a given object is a function or method defined within a
 * Particle derived type. If so, return the type pointer,
 * NULL otherwise.
 */
CAPI_FUNC(MxParticleType*) MxParticleType_FromFunction(PyObject *e);

HRESULT _MxTimeEvent_Init(PyObject *m);

#endif /* SRC_MDCORE_INCLUDE_MXPARTICLEEVENT_H_ */
