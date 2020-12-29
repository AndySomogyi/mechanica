/*
 * MxColorMapper.h
 *
 *  Created on: Dec 27, 2020
 *      Author: andy
 */

#ifndef SRC_RENDERING_MXCOLORMAPPER_H_
#define SRC_RENDERING_MXCOLORMAPPER_H_

#include <rendering/NOMStyle.hpp>
#include <Magnum/Magnum.h>
#include <Magnum/Math/Color.h>


struct MxColorMapper
{
    ColorMapperFunc map;
    
    float min_val;
    float max_val;
};



MxColorMapper *MxColorMapper_New(struct MxParticleType *partType,
                                 const char* speciesName,
                                 const char* name, float min, float max);

/**
 * Makes a new color map.
 * the first arg, args should be a MxParticleType object.
 *
 * since this is a style, presently this method will not set any error
 * conditions, but will set a warnign, and return null on failure.
 */
MxColorMapper *MxColorMapper_New(PyObject *args, PyObject *kwargs);




#endif /* SRC_RENDERING_MXCOLORMAPPER_H_ */
