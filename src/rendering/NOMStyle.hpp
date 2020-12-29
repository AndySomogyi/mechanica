/*
 * NOMStyle.hpp
 *
 *  Created on: Jul 29, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_NOMSTYLE_HPP_
#define SRC_RENDERING_NOMSTYLE_HPP_

#include <NOMStyle.h>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>

typedef Magnum::Color4 (*ColorMapperFunc)(struct MxColorMapper *mapper, struct MxParticle *p);

struct NOMStyle : public PyObject
{
    Magnum::Color3 color;
    uint32_t flags;
    
    struct MxColorMapper *mapper;
    
    ColorMapperFunc mapper_func;
    
    inline Magnum::Color4 map_color(struct MxParticle *p) {
        if(mapper_func) {
            return mapper_func(mapper, p);
        }
        return Magnum::Color4{color, 1};
    };
};

CAPI_FUNC(NOMStyle*) NOMStyle_NewEx(const Magnum::Color3& color, uint32_t flags = StyleFlags::STYLE_VISIBLE);


HRESULT _NOMStyle_init(PyObject *m);

#endif /* SRC_RENDERING_NOMSTYLE_HPP_ */
