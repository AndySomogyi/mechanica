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
    IS_3DNOW              = 1ul << 0,
    IS_3DNOWEXT           = 1ul << 1,
    IS_ABM                = 1ul << 2,
    IS_ADX                = 1ul << 3,
    IS_AES                = 1ul << 4,
    IS_AVX                = 1ul << 5,
    IS_AVX2               = 1ul << 6,
    IS_AVX512CD           = 1ul << 7,
    IS_AVX512ER           = 1ul << 8,
    IS_AVX512F            = 1ul << 9,
    IS_AVX512PF           = 1ul << 10,
    IS_BMI1               = 1ul << 11,
    IS_BMI2               = 1ul << 12,
    IS_CLFSH              = 1ul << 13,
    IS_CMPXCHG16B         = 1ul << 14,
    IS_CX8                = 1ul << 15,
    IS_ERMS               = 1ul << 16,
    IS_F16C               = 1ul << 17,
    IS_FMA                = 1ul << 18,
    IS_FSGSBASE           = 1ul << 19,
    IS_FXSR               = 1ul << 20,
    IS_HLE                = 1ul << 21,
    IS_INVPCID            = 1ul << 23,
    IS_LAHF               = 1ul << 24,
    IS_LZCNT              = 1ul << 25,
    IS_MMX                = 1ul << 26,
    IS_MMXEXT             = 1ul << 27,
    IS_MONITOR            = 1ul << 28,
    IS_MOVBE              = 1ul << 28,
    IS_MSR                = 1ul << 29,
    IS_OSXSAVE            = 1ul << 30,
    IS_PCLMULQDQ          = 1ul << 31,
    IS_POPCNT             = 1ul << 32,
    IS_PREFETCHWT1        = 1ul << 33,
    IS_RDRAND             = 1ul << 34,
    IS_RDSEED             = 1ul << 35,
    IS_RDTSCP             = 1ul << 36,
    IS_RTM                = 1ul << 37,
    IS_SEP                = 1ul << 38,
    IS_SHA                = 1ul << 39,
    IS_SSE                = 1ul << 40,
    IS_SSE2               = 1ul << 41,
    IS_SSE3               = 1ul << 42,
    IS_SSE41              = 1ul << 43,
    IS_SSE42              = 1ul << 44,
    IS_SSE4a              = 1ul << 45,
    IS_SSSE3              = 1ul << 46,
    IS_SYSCALL            = 1ul << 47,
    IS_TBM                = 1ul << 48,
    IS_XOP                = 1ul << 49,
    IS_XSAVE              = 1ul << 50,
};

PyObject *MxInstructionSetFeatruesDict();

PyObject *MxCompileFlagsDict();

uint64_t MxInstructionSetFeatures();


HRESULT _MxUtil_init(PyObject *m);



#endif /* SRC_MXUTIL_H_ */
