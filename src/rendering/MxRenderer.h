/*
 * MxRenderer.h
 *
 *  Created on: Apr 15, 2020
 *      Author: andy
 */

#ifndef SRC_MXRENDERER_H_
#define SRC_MXRENDERER_H_

#include <mechanica_private.h>

struct MxRenderer 
{
};

enum MxRenderer_Kind {
    RENDERER_WINDOWED               = 1 << 0,
    RENDERER_HEADLESS             = 1 << 1,
    
    RENDERER_WINDOWED_MAC           = (1 << 2) | (1 << 0),
    RENDERER_HEADLESS_MAC         = (1 << 2) | (1 << 1),
    
    RENDERER_WINDOWED_EGL           = (1 << 2) | (1 << 0),
    RENDERER_HEADLESS_EGL         = (1 << 2) | (1 << 1),
   
    RENDERER_WINDOWED_WINDOWS       = (1 << 2) | (1 << 0),
    RENDERER_HEADLESS_WINDOWS     = (1 << 2) | (1 << 1),
    
    RENDERER_WINDOWED_GLX           = (1 << 2) | (1 << 0),
    RENDERER_HEADLESS_GLX         = (1 << 2) | (1 << 1),
};

uint32_t MxRenderer_AvailableRenderers();

/**
 * The the particle type type
 */
CAPI_DATA(PyTypeObject) MxRenderer_Type;

#endif /* SRC_MXRENDERER_H_ */
