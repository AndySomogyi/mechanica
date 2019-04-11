/*
 * CylinderModel.h
 *
 *  Created on: Sep 20, 2018
 *      Author: andy
 */

#ifndef _INCLUDED_MX_CYLINDERMODEL_H_
#define _INCLUDED_MX_CYLINDERMODEL_H_

#include "MxModel.h"
#include <LangevinPropagator.h>

struct MxCylinderModel : public MxModel {

    enum VolumeForce {
        ConstantVolume, ConstantPressure
    };

    MxCylinderModel();

    HRESULT loadModel(const char* fileName);

    float minTargetVolume();
    float maxTargetVolume();
    float targetVolume();
    float targetVolumeLambda();
    void setTargetVolume(float targetVolume);
    void setTargetVolumeLambda(float targetVolumeLambda);
    
    float stdSurfaceTension();
    void setStdSurfaceTension(float val);
    float stdSurfaceTensionMin();
    float stdSurfaceTensionMax();
    
    float growSurfaceTension();
    void setGrowStdSurfaceTension(float val);
    float growSurfaceTensionMin();
    float growSurfaceTensionMax();

    float minTargetArea();
    float maxTargetArea();
    float targetArea();
    float targetAreaLambda();
    void setTargetArea(float targetArea);
    void setTargetAreaLambda(float targetAreaLambda);

    void testEdges();

    HRESULT applyT1Edge2TransitionToSelectedEdge();

    HRESULT applyT2PolygonTransitionToSelectedPolygon();

    HRESULT applyT3PolygonTransitionToSelectedPolygon();

    void loadAssImpModel(const char* fileName);

    HRESULT changePolygonTypes();

    HRESULT activateAreaConstraint();

    /**
      * The state vector is a vector of elements that are defined by
      * differential equations (rate rules) or independent floating species
      * are defined by reactions.
      *
      * To get the ids of the state vector elements, use getStateVectorId.
      *
      * copies the internal model state vector into the provided
      * buffer.
      *
      * @param[out] stateVector: a buffer to copy the state vector into, if NULL,
      *         return the size required.
      *
      * @param[out] count: the number of items coppied into the provided buffer, if
      *         stateVector is NULL, returns the length of the state vector.
      */
     virtual HRESULT getStateVector(float *stateVector, uint32_t *count);

     /**
      * sets the internal model state to the provided packed state vector.
      *
      * @param[in] an array which holds the packed state vector, must be
      *         at least the size returned by getStateVector.
      *
      * @return the number of items copied from the state vector, negative
      *         on failure.
      */
     virtual HRESULT setStateVector(const float *stateVector);

     /**
      * the state vector y is the rate rule values and floating species
      * concentrations concatenated. y is of length numFloatingSpecies + numRateRules.
      *
      * The state vector is packed such that the first n raterule elements are the
      * values of the rate rules, and the last n floatingspecies are the floating
      * species values.
      *
      * @param[in] time current simulator time
      * @param[in] y state vector, must be either null, or have a size of that
      *         speciefied by getStateVector. If y is null, then the model is
      *         evaluated using its current state. If y is not null, then the
      *         y is considered the state vector.
      * @param[out] dydt calculated rate of change of the state vector, if null,
      *         it is ignored.
      */
     virtual HRESULT getStateVectorRate(float time, const float *y, float* dydt=0);
};


/**
 * The type object for a MxSymbol.
 */
MxAPI_DATA(PyTypeObject) *MxCylinderModel_Type;

HRESULT MxCylinderModel_init(PyObject *m);


#endif /* _INCLUDED_MX_CYLINDERMODEL_H_ */
