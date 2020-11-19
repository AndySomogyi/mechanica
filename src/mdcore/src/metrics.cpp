/*
 * metrics.cpp
 *
 *  Created on: Nov 18, 2020
 *      Author: andy
 */

#include <metrics.h>
#include <engine.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Vector3.h>
#include "MxConvert.hpp"


HRESULT MxCalculatePressure(FPTYPE *_origin,
                                       FPTYPE radius,
                                       const std::vector<unsigned> typeIds,
                            FPTYPE *tensor) {
    Magnum::Vector3 origin = Magnum::Vector3::from(_origin);
    
    
    
}

/**
 * converts cartesian to spherical in global coord space.
 * createsa a numpy array.
 */
PyObject* MPyCartesianToSpherical(const Magnum::Vector3& postion,
                                  const Magnum::Vector3& origin) {
    return mx::cast(MxCartesianToSpherical(postion, origin));
}


/**
 * converts cartesian to spherical, writes spherical
 * coords in to result array.
 */
Magnum::Vector3 MxCartesianToSpherical(const Magnum::Vector3& pos,
                                       const Magnum::Vector3& origin) {
    Magnum::Vector3 vec = pos - origin;
    
    float radius = vec.length();
    float theta = std::atan2(vec.y(), vec.x());
    float phi = std::acos(vec.z() / radius);
    return Magnum::Vector3{radius, theta, phi};
}
