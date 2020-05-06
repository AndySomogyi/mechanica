
#include "MxGlfwApplication.h"

#include <cstring>
#include <tuple>
#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/StridedArrayView.h>
#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/String.h>
#include <Corrade/Utility/Unicode.h>

#include "Magnum/ImageView.h"
#include "Magnum/PixelFormat.h"
#include "Magnum/Math/ConfigurationValue.h"
#include "Magnum/Platform/ScreenedApplication.hpp"
#include "Magnum/Platform/Implementation/DpiScaling.h"


#define MXGLFW_ERROR() { \
    const char* glfwErrorDesc = NULL; \
    glfwGetError(&glfwErrorDesc); \
    throw std::domain_error(std::string("GLFW Error in ") + MX_FUNCTION + ": " +  glfwErrorDesc); \
}

#define MXGLFW_CHECK() { \
    const char* glfwErrorDesc = NULL; \
    int ret = glfwGetError(&glfwErrorDesc); \
    return ret == 0 ? S_OK : mx_error(ret, glfwErrorDesc); \
}


static GlfwApplication::Configuration confconf(const MxSimulator::Config &conf) {
    GlfwApplication::Configuration res;

    res.setSize(conf.size(), conf.dpiScaling());
    res.setTitle(conf.title());
    res.setWindowFlags(GlfwApplication::Configuration::WindowFlag::Resizable);

    return res;
}


MxGlfwApplication::MxGlfwApplication(const Arguments &args) :
        GlfwApplication{args, NoCreate}
{
}



HRESULT MxGlfwApplication::pollEvents()
{
    glfwPollEvents();
    MXGLFW_CHECK();
}

HRESULT MxGlfwApplication::waitEvents()
{
    glfwWaitEvents();
    MXGLFW_CHECK();
}

HRESULT MxGlfwApplication::waitEventsTimeout(double timeout)
{
    glfwWaitEventsTimeout(timeout);
    MXGLFW_CHECK();
}

HRESULT MxGlfwApplication::postEmptyEvent()
{
    glfwPostEmptyEvent();
    MXGLFW_CHECK();
}

void MxGlfwApplication::drawEvent()
{
}



HRESULT MxGlfwApplication::createContext(
        const MxSimulator::Config &conf)
{
    bool b = tryCreate(confconf(conf));


    if(b) {

        _win = new MxGlfwWindow(this->window());
    }
    return b ? S_OK : E_FAIL;
}

MxGlfwWindow* MxGlfwApplication::getWindow()
{
    return _win;
}

HRESULT MxGlfwApplication::setSwapInterval(int si)
{
    glfwSwapInterval(si);
    MXGLFW_CHECK();
}
