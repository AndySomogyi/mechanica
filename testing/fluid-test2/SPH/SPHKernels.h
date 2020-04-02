/*
    This file is part of Mechanica.

    Based on Magnum example

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2019 — Nghia Truong <nghiatruong.vn@gmail.com>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
 */


#pragma once

#include <Magnum/Magnum.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Math/Vector3.h>

using namespace Magnum;


class Poly6Kernel {
    public:
        void setRadius(const Float radius) {
            _radius = radius;
            _radiusSqr = _radius*_radius;
            _k = 315.0f/(64.0f*Constants::pi()*Math::pow(_radius, 9.0f));
            _W0 = W(0.0f);
        }

        Float W(const Float r) const {
            const Float r2 = r * r;
            return r2 <= _radiusSqr ? Math::pow(_radiusSqr - r2, 3.0f)*_k : 0.0f;
        }

        Float W(const Vector3& r) const {
            const auto r2 = r.dot();
            return r2 <= _radiusSqr ? Math::pow(_radiusSqr - r2, 3.0f)*_k : 0.0f;
        }

        Float W0() const { return _W0; }

    private:
        Float _radius;
        Float _radiusSqr;
        Float _k;
        Float _W0;
};

class SpikyKernel {
    public:
        void setRadius(const Float radius) {
            _radius = radius;
            _radiusSqr = _radius * _radius;
            _l = -45.0f/(Constants::pi()*Math::pow(_radius, 6.0f));
        }

        Vector3 gradW(const Vector3& r) const {
            Vector3 res{0.0f};
            const Float r2 = r.dot();
            if(r2 <= _radiusSqr && r2 > 1.0e-12f) {
                const Float rl = Math::sqrt(r2);
                const Float hr = _radius - rl;
                const Float hr2 = hr * hr;
                res = _l*hr2*(r/rl);
            }

            return res;
        }

    protected:
        Float _radius;
        Float _radiusSqr;
        Float _l;
};

class SPHKernels {
    public:
        explicit SPHKernels(Float kernelRadius) {
            _poly6.setRadius(kernelRadius);
            _spiky.setRadius(kernelRadius);
        }

        Float W0() const { return _poly6.W0(); }
        Float W(const Vector3& r) const { return _poly6.W(r); }
        Vector3 gradW(const Vector3& r) const { return _spiky.gradW(r); }

    private:
        Poly6Kernel _poly6;
        SpikyKernel _spiky;
};


