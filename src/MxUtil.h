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

enum Mx_InstructionSet : std::int64_t {
    IS_3DNOW              = 1ll << 0,
    IS_3DNOWEXT           = 1ll << 1,
    IS_ABM                = 1ll << 2,
    IS_ADX                = 1ll << 3,
    IS_AES                = 1ll << 4,
    IS_AVX                = 1ll << 5,
    IS_AVX2               = 1ll << 6,
    IS_AVX512CD           = 1ll << 7,
    IS_AVX512ER           = 1ll << 8,
    IS_AVX512F            = 1ll << 9,
    IS_AVX512PF           = 1ll << 10,
    IS_BMI1               = 1ll << 11,
    IS_BMI2               = 1ll << 12,
    IS_CLFSH              = 1ll << 13,
    IS_CMPXCHG16B         = 1ll << 14,
    IS_CX8                = 1ll << 15,
    IS_ERMS               = 1ll << 16,
    IS_F16C               = 1ll << 17,
    IS_FMA                = 1ll << 18,
    IS_FSGSBASE           = 1ll << 19,
    IS_FXSR               = 1ll << 20,
    IS_HLE                = 1ll << 21,
    IS_INVPCID            = 1ll << 23,
    IS_LAHF               = 1ll << 24,
    IS_LZCNT              = 1ll << 25,
    IS_MMX                = 1ll << 26,
    IS_MMXEXT             = 1ll << 27,
    IS_MONITOR            = 1ll << 28,
    IS_MOVBE              = 1ll << 28,
    IS_MSR                = 1ll << 29,
    IS_OSXSAVE            = 1ll << 30,
    IS_PCLMULQDQ          = 1ll << 31,
    IS_POPCNT             = 1ll << 32,
    IS_PREFETCHWT1        = 1ll << 33,
    IS_RDRAND             = 1ll << 34,
    IS_RDSEED             = 1ll << 35,
    IS_RDTSCP             = 1ll << 36,
    IS_RTM                = 1ll << 37,
    IS_SEP                = 1ll << 38,
    IS_SHA                = 1ll << 39,
    IS_SSE                = 1ll << 40,
    IS_SSE2               = 1ll << 41,
    IS_SSE3               = 1ll << 42,
    IS_SSE41              = 1ll << 43,
    IS_SSE42              = 1ll << 44,
    IS_SSE4a              = 1ll << 45,
    IS_SSSE3              = 1ll << 46,
    IS_SYSCALL            = 1ll << 47,
    IS_TBM                = 1ll << 48,
    IS_XOP                = 1ll << 49,
    IS_XSAVE              = 1ll << 50,
};

PyObject *MxInstructionSetFeatruesDict(PyObject *o);

PyObject *MxCompileFlagsDict(PyObject *o);

uint64_t MxInstructionSetFeatures();
    
    
CAPI_FUNC(double) MxWallTime();

CAPI_FUNC(double) MxCPUTime();


class WallTime {
public:
    WallTime();
    ~WallTime();
    double start;
};
    
    


HRESULT _MxUtil_init(PyObject *m);



#endif /* SRC_MXUTIL_H_ */
