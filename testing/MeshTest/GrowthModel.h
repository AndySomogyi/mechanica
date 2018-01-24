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

    GrowthModel();

    /**
     * Evaluate the force functions,
     */
    HRESULT calcForce() ;


    HRESULT cellAreaForce(CellPtr cell);

    HRESULT cellVolumeForce(CellPtr cell);

    float minTargetArea;
    float maxTargetArea;
    float targetArea;

    float minTargetVolume;
    float maxTargetVolume;
    float targetVolume;

    void testEdges();


    void loadSheetModel();

    void loadSimpleSheetModel();

    void loadCubeModel();

    void loadMonodisperseVoronoiModel();

    virtual HRESULT getForces(float time, uint32_t len, const Vector3 *pos, Vector3 *force);

    virtual HRESULT getAccelerations(float time, uint32_t len, const Vector3 *pos, Vector3 *acc);

    virtual HRESULT getMasses(float time, uint32_t len, float *masses);

    virtual HRESULT getPositions(float time, uint32_t len, Vector3 *pos);

    virtual HRESULT setPositions(float time, uint32_t len, const Vector3 *pos);
};


#endif /* TESTING_GROWTH1_GROWTHMODEL_H_ */
