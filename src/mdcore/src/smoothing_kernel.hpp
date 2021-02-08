/*
 * smoothing_kernel.hpp
 *
 *  Created on: Jan 27, 2021
 *      Author: andy
 */

#ifndef SRC_MDCORE_SRC_SMOOTHING_KERNEL_HPP_
#define SRC_MDCORE_SRC_SMOOTHING_KERNEL_HPP_

#include "mdcore_config.h"
#include <cmath>

#include <immintrin.h>


// faster than  1.0f/std::sqrt, but with little accuracy.
MX_ALWAYS_INLINE float qsqrt(const float f)
{
    __m128 temp = _mm_set_ss(f);
    temp = _mm_rsqrt_ss(temp);
    return 1.0 / _mm_cvtss_f32(temp);
}

MX_ALWAYS_INLINE float w_cubic_spline(float r2, float h) {
    float r = qsqrt(r2);
    float x = r/h;
    float y;
    
    if(x < 1.f) {
        float x2 = x * x;
        y = 1.f - (3.f / 2.f) * x2 + (3.f / 4.f) * x2 * x;
    }
    else if(x >= 1.f && x < 2.f) {
        float arg = 2.f - x;
        y = (1.f / 4.f) * arg * arg * arg;
    }
    else {
        y = 0.f;
    }
    
    return y / (M_PI * h * h * h);
}

MX_ALWAYS_INLINE float grad_w_cubic_spline(float r2, float h) {
    float r = qsqrt(r2);
    float x = r/h;
    float y;
    
    if(x < 1.f) {
        y = (9.f / 4.f) * x * x  - (3.f ) * x;
    }
    else if(x >= 1.f && x < 2.f) {
        float arg = 2.f - x;
        y = -(3.f / 4.f) * arg * arg;
    }
    else {
        y = 0.f;
    }
    
    return y / (M_PI * h * h * h * h);
}

MX_ALWAYS_INLINE float W(float r2, float h) { return w_cubic_spline(r2, h); };

MX_ALWAYS_INLINE float grad_W(float r2, float h) { return grad_w_cubic_spline(r2, h); };








#endif /* SRC_MDCORE_SRC_SMOOTHING_KERNEL_HPP_ */
