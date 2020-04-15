/*
 * MxWindowlessApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <rendering/MxWindowlessApplication.h>

#include <iostream>




static Magnum::Platform::WindowlessApplication::Configuration
    config(const MxApplication::Configuration) {

    Magnum::Platform::WindowlessApplication::Configuration result;


#ifdef MX_LINUX
    result.clearFlags(Magnum::Platform::WindowlessApplication::Configuration::Flag::ForwardCompatible);
#endif

    return result;
}


MxWindowlessApplication::MxWindowlessApplication(
        int argc, char** argv, const MxApplication::Configuration& conf) :
        Magnum::Platform::WindowlessApplication({argc, argv}, config(conf)) {

}

MxWindowlessApplication::~MxWindowlessApplication()
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}


