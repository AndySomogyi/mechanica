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

enum Mx_InstructionSet {
    IS_3DNOW              = 1 << 0,
    IS_3DNOWEXT           = 1 << 1,
    IS_ABM                = 1 << 2,
    IS_ADX                = 1 << 3,
    IS_AES                = 1 << 4,
    IS_AVX                = 1 << 5,
    IS_AVX2               = 1 << 6,
    IS_AVX512CD           = 1 << 7,
    IS_AVX512ER           = 1 << 8,
    IS_AVX512F            = 1 << 9,
    IS_AVX512PF           = 1 << 10,
    IS_BMI1               = 1 << 11,
    IS_BMI2               = 1 << 12,
    IS_CLFSH              = 1 << 13,
    IS_CMPXCHG16B         = 1 << 14,
    IS_CX8                = 1 << 15,
    IS_ERMS               = 1 << 16,
    IS_F16C               = 1 << 17,
    IS_FMA                = 1 << 18,
    IS_FSGSBASE           = 1 << 19,
    IS_FXSR               = 1 << 20,
    IS_HLE                = 1 << 21,
    IS_INVPCID            = 1 << 23,
    IS_LAHF               = 1 << 24,
    IS_LZCNT              = 1 << 25,
    IS_MMX                = 1 << 26,
    IS_MMXEXT             = 1 << 27,
    IS_MONITOR            = 1 << 28,
    IS_MOVBE              = 1 << 28,
    IS_MSR                = 1 << 29,
    IS_OSXSAVE            = 1 << 30,
    IS_PCLMULQDQ          = 1 << 31,
    IS_POPCNT             = 1 << 32,
    IS_PREFETCHWT1        = 1 << 33,
    IS_RDRAND             = 1 << 34,
    IS_RDSEED             = 1 << 35,
    IS_RDTSCP             = 1 << 36,
    IS_RTM                = 1 << 37,
    IS_SEP                = 1 << 38,
    IS_SHA                = 1 << 39,
    IS_SSE                = 1 << 40,
    IS_SSE2               = 1 << 41,
    IS_SSE3               = 1 << 42,
    IS_SSE41              = 1 << 43,
    IS_SSE42              = 1 << 44,
    IS_SSE4a              = 1 << 45,
    IS_SSSE3              = 1 << 46,
    IS_SYSCALL            = 1 << 47,
    IS_TBM                = 1 << 48,
    IS_XOP                = 1 << 49,
    IS_XSAVE              = 1 << 50,
};

PyObject *MxInstructionSetFeatruesDict();

PyObject *MxCompileFlagsDict();

uint64_t MxInstructionSetFeatures();


HRESULT _MxUtil_init(PyObject *m);



#endif /* SRC_MXUTIL_H_ */
