/*
 * metrics.h
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#ifndef SRC_MDCORE_INCLUDE_METRICS_H_
#define SRC_MDCORE_INCLUDE_METRICS_H_

#include "platform.h"
#include "mdcore_config.h"
#include "Magnum/Magnum.h"
#include "Magnum/Math/Vector3.h"

/**
 * @origin [in] origin of the sphere where we will comptute
 * the local pressure tensor.
 * @radius [in] include all partices a given radius in calculation. 
 * @typeIds [in] vector of type ids to indlude in calculation,
 * if empty, includes all particles.
 * @tensor [out] result vector, writes a 3x3 matrix in a row-major in the given
 * location.
 */
CAPI_FUNC(HRESULT) MxCalculatePressure(FPTYPE *origin,
                                       FPTYPE radius,
                                       const std::vector<unsigned> typeIds,
                                       FPTYPE *tensor);


/**
 * converts cartesian to spherical in global coord space.
 * createsa a numpy array.
 */
PyObject* MPyCartesianToSpherical(const Magnum::Vector3& postion,
                                             const Magnum::Vector3& origin);


/**
 * converts cartesian to spherical, writes spherical
 * coords in to result array.
 */
Magnum::Vector3 MxCartesianToSpherical(const Magnum::Vector3& postion,
                                          const Magnum::Vector3& origin);





#endif /* SRC_MDCORE_INCLUDE_METRICS_H_ */
