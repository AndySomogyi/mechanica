/*
 * MxParticleList.h
 *
 *  Created on: Nov 23, 2020
 *      Author: andy
 */

#ifndef _MDCORE_MXPARTICLELIST_H_
#define _MDCORE_MXPARTICLELIST_H_

#include "c_port.h"

enum ParticleListFlags {
    // list owns the data the MxParticleList::parts
    PARTICLELIST_OWNDATA = 1 << 0,
    
    // list supports insertion / deletion
    PARTICLELIST_MUTABLE = 1 << 1,
    
    // list owns it's own data, it was allocated.
    PARTICLELIST_OWNSELF = 1 << 2,
};

/** The #potential structure. */
struct MxParticleList : PyObject {
    int32_t *parts;
    int32_t nr_parts;
    int32_t size_parts;
    uint16_t flags;
    
    // particle list designed to be alloc'ed in single block,
    // this initializes this section.
    void init();
    
    // frees the memory associated with the parts list.
    void free();
    
    // inserts the given id into the list, returns the
    // index of the item. 
    uint16_t insert(int32_t item);
    
    // looks for the item with the given id and deletes it form the
    // list
    uint16_t remove(int32_t id);
    
};

/**
 * Tries to create a new list from a python list.
 * if the python list is a list of MxPyParticle, creates a new particle
 * list from them.
 *
 * If the list is already a MxParticleList, increments it and returns
 * it.
 *
 * The pyobject can be a single MxParticleHandle, if so, this constructs a list
 * containing a single element.
 *
 * The caller is owns a new reference to the list, and is responsible for
 * freeing it.
 *
 * Returns NULL if list is not a list or doesnt contain particles. 
 */
CAPI_FUNC(MxParticleList*) MxParticleList_FromPyObject(PyObject *obj);


/**
 * new list with initial capacity but no items.
 */
CAPI_FUNC(MxParticleList*) MxParticleList_New(uint16_t init_size,
                                              uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);

/**
 * 
 */
CAPI_FUNC(int) MxParticleList_Check(const PyObject *obj);

/**
 *
 */
CAPI_FUNC(MxParticleList*) MxParticleList_Copy(const PyObject *obj);

/**
 * New list, steals the data.
 */
CAPI_FUNC(MxParticleList*) MxParticleList_NewFromData(uint16_t nr_parts, int32_t *parts);


/**
 * creates a new, packed particle list.
 * initial flags are PARTICLELIST_OWNDATA | PARTICLELIST_OWNSELF
 */
CAPI_FUNC(PyObject*) MxParticleList_Pack(Py_ssize_t n, ...);


/**
 * The type of each individual particle.
 */
CAPI_DATA(PyTypeObject) MxParticleList_Type;

HRESULT _MxParticleList_init(PyObject *m);





#endif /* SRC_MDCORE_SRC_MXPARTICLELIST_H_ */
