/*
 * GrowthModel.h
 *
 *  Created on: Oct 13, 2017
 *      Author: andy
 */

#ifndef TESTING_GROWTH1_GROWTHMODEL_H_
#define TESTING_GROWTH1_GROWTHMODEL_H_

#include "MxModel.h"

struct GrowthModel : public MxModel {

    enum VolumeForce {
        ConstantVolume, ConstantPressure
    };

    GrowthModel();


    HRESULT loadModel();

    float minTargetVolume();
    float maxTargetVolume();
    float targetVolume();
    float targetVolumeLambda();

    void setTargetVolume(float targetVolume);

    void setTargetVolumeLambda(float targetVolumeLambda);

    float harmonicBondStrength;

    float pressure = 0;
    float pressureMax;
    float pressureMin;

    float stdSurfaceTension();
    void setStdSurfaceTension(float val);
    float stdSurfaceTensionMin();
    float stdSurfaceTensionMax();

    void testEdges();

    HRESULT applyT1Edge2TransitionToSelectedEdge();

    HRESULT applyT2PolygonTransitionToSelectedPolygon();

    HRESULT applyT3PolygonTransitionToSelectedPolygon();

    void loadSheetModel();

    void loadTwoModel();

    void loadSimpleSheetModel();

    void loadCubeModel();

    void loadMonodisperseVoronoiModel();

    void loadAssImpModel();

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


#endif /* TESTING_GROWTH1_GROWTHMODEL_H_ */
