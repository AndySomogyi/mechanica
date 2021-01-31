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

MX_ALWAYS_INLINE float w_cubic_spline(float r, float h) {
    float x = std::abs(r)/h;
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

MX_ALWAYS_INLINE float grad_w_cubic_spline(float r, float h) {
    float x = std::abs(r)/h;
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

MX_ALWAYS_INLINE float W(float r, float h) { return w_cubic_spline(r, h); };

MX_ALWAYS_INLINE float grad_W(float r, float h) { return grad_w_cubic_spline(r, h); };





#endif /* SRC_MDCORE_SRC_SMOOTHING_KERNEL_HPP_ */
