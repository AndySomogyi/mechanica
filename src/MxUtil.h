/*
 * MxParticles.h
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifndef SRC_MXUTIL_H_
#define SRC_MXUTIL_H_

#include "mechanica_private.h"

#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Vector3.h>


#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <limits>
#include <type_traits>

enum MxPointsType {
    Sphere,
    SolidSphere,
    Disk,
    SolidCube,
    Cube,
    Ring
};

Magnum::Color3 Color3_Parse(const std::string &str);




PyObject* MxRandomPoints(PyObject *m, PyObject *args, PyObject *kwargs);

PyObject* MxPoints(PyObject *m, PyObject *args, PyObject *kwargs);

extern const char* MxColor3Names[];

HRESULT Mx_Icosphere(const int subdivisions, float phi0, float phi1,
                     std::vector<Magnum::Vector3> &verts,
                     std::vector<int32_t> &inds);

namespace mx {
    template<class T>
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp = 2)
    {
        // the machine epsilon has to be scaled to the magnitude of the values used
        // and multiplied by the desired precision in ULPs (units in the last place)
        return std::fabs(x-y) <= std::numeric_limits<T>::epsilon() * std::fabs(x+y) * ulp
        // unless the result is subnormal
        || std::fabs(x-y) < std::numeric_limits<T>::min();
    }
}


HRESULT _MxUtil_init(PyObject *m);



#endif /* SRC_MXUTIL_H_ */
