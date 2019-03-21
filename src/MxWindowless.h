/*
 * MxWindowless.h
 *
 *  Created on: Mar 21, 2019
 *      Author: andy
 */

#ifndef SRC_MXWINDOWLESS_H_
#define SRC_MXWINDOWLESS_H_


#include <Mechanica.h>
#include <Magnum/GL/Context.h>

#if defined(MX_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
    typedef Magnum::Platform::WindowlessCglContext WindowlessContext;
    typedef Magnum::Platform::WindowlessCglApplication WindowlessApplication;
#elif defined(MX_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"
    typedef Magnum::Platform::WindowlessEglContext WindowlessContext;
    typedef Magnum::Platform::WindowlessEglApplication WindowlessApplication;
#elif defined(MX_WINDOWS)
#include "Magnum/Platform/WindowlessWglApplication.h"
#else
#error no windowless application available on this platform
#endif



#endif /* SRC_MXWINDOWLESS_H_ */
