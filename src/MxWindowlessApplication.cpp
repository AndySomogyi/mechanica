/*
 * MxWindowlessApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <MxWindowlessApplication.h>

#include <iostream>

class ProxyWindowlessApplication: public Magnum::Platform::WindowlessApplication {
public:


    ProxyWindowlessApplication(const Magnum::Platform::WindowlessApplication::Arguments &args,
            const Magnum::Platform::WindowlessApplication::Configuration &conf) :
                Magnum::Platform::WindowlessApplication(args, conf) {
    }

    virtual int exec() override { return 0; };
    virtual ~ProxyWindowlessApplication() {};

};


static Magnum::Platform::WindowlessApplication::Configuration
    config(const MxApplication::Configuration) {

    Magnum::Platform::WindowlessApplication::Configuration result;

    return result;
}

static Magnum::Platform::WindowlessApplication::Arguments args() {
    int argc = 0;
    char** argv = nullptr;
    return Magnum::Platform::WindowlessApplication::Arguments(argc, argv);
}

MxWindowlessApplication::MxWindowlessApplication(
        int argc, char** argv, const MxApplication::Configuration& conf )
{
    Magnum::Platform::WindowlessApplication::Configuration magnumConf;

#ifdef MX_LINUX
    magnumConf.clearFlags(Magnum::Platform::WindowlessApplication::Configuration::Flag::ForwardCompatible);
#endif

    app = new ProxyWindowlessApplication({argc, argv}, magnumConf);
}

MxWindowlessApplication::~MxWindowlessApplication()
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
    delete app;
}


