/*
 * MxWindowlessApplication.h
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#ifndef SRC_MXWINDOWLESSAPPLICATION_H_
#define SRC_MXWINDOWLESSAPPLICATION_H_

#include "MxApplication.h"
#include <Magnum/GL/Context.h>


#if defined(MX_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
#elif defined(MX_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"
#elif defined(MX_WINDOWS)
#include "Magnum/Platform/WindowlessWglApplication.h"
#else
#error no windowless application available on this platform
#endif


class ProxyWindowlessApplication;

struct  MxWindowlessApplication : public MxApplication
{
public:

    MxWindowlessApplication() = delete;

    MxWindowlessApplication(int argc, char** argv, const MxApplication::Configuration& conf);

    virtual ~MxWindowlessApplication();



private:

    ProxyWindowlessApplication *app;

};

#endif /* SRC_MXWINDOWLESSAPPLICATION_H_ */
