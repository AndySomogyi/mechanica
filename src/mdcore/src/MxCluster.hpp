/*
 * MxCluster.h
 *
 *  Created on: Aug 28, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_MXCLUSTER_H_
#define SRC_MDCORE_SRC_MXCLUSTER_H_

#include <MxParticle.h>

struct MxCluster : MxParticle
{
};


/**
 * pointer the cluster  type,
 *
 * this is a derived type, so this is actually the 2nd element in
 * the gloabl types array.
 */
CAPI_DATA(MxParticleType*) MxCluster_TypePtr;

/**
 * shim object (or maybe a 'closure') to package up a cluster, and
 * particle type to act as a particle constructor in a cluster.
 */
struct MxParticleConstructor;

/**
 * creates a particle constructor (internal method).
 *
 * creates a python descrptor that gets stuffed in the cluster dict
 * that makes new partice instances in the particle.
 */
PyObject *MxClusterParticleCtor_New(
    MxParticleType *clusterType, MxParticleType *containedParticleType);

PyObject *makeThing();

CAPI_FUNC(int) MxCluster_Check(PyObject *p);

/**
 * adds an existing particle to the cluster.
 */
CAPI_FUNC(int) MxCluster_AddParticle(struct MxCluster *cluster, struct MxParticle *part);


/**
 * Computes the aggregate quanties such as total mass, position, acceleration, etc...
 * from the contained particles. 
 */
CAPI_FUNC(int) MxCluster_ComputeAggregateQuantities(struct MxCluster *cluster);

/**
 * creates a new particle, and adds it to the cluster.
 */
CAPI_FUNC(PyObject*) MxCluster_CreateParticle(PyObject *self,
    PyObject* particleType, PyObject *args, PyObject *kwargs);


/**
 * sequence methods for the cluster, should proably not be public,
 * but these are defined in cluster.cpp, as there is already way too much
 * stuff in particle.cpp
 */
CAPI_DATA(PySequenceMethods) MxCluster_Sequence;


HRESULT MxClusterType_Init(MxParticleType *self, PyObject *_dict);

/**
 * internal function to initalize the particle and particle types
 */
HRESULT _MxCluster_init(PyObject *m);


#endif /* SRC_MDCORE_SRC_MXCLUSTER_H_ */
