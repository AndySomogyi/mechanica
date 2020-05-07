
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

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/Animation/Easing.h>


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

void MxGlfwApplication::simulationStep() {
    static Float offset = 0.0f;
    if(_dynamicBoundary) {
        /* Change fluid boundary */
        static Float step = 2.0e-3f;
        if(_boundaryOffset > 1.0f || _boundaryOffset < 0.0f) {
            step *= -1.0f;
        }
        _boundaryOffset += step;
        offset = Math::lerp(0.0f, 0.5f, Animation::Easing::quadraticInOut(_boundaryOffset));
    }


    engineStep();

    currentStep += 1;
}

void MxGlfwApplication::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);



    /* Pause simulation if the mouse was pressed (camera is moving around).
       This avoid freezing GUI while running the simulation */
    if(!_pausedSimulation && !_mousePressed) {
        /* Adjust the substep number to maximize CPU usage each frame */
        const Float lastAvgStepTime = _timeline.previousFrameDuration()/Float(_substeps);
        const Int newSubsteps = lastAvgStepTime > 0 ? Int(1.0f/60.0f/lastAvgStepTime) + 1 : 1;
        if(Math::abs(newSubsteps - _substeps) > 1) _substeps = newSubsteps;

        for(Int i = 0; i < _substeps; ++i) simulationStep();
    }


    /* Draw particles */
    _ren->draw();



    swapBuffers();
    _timeline.nextFrame();

    /* Run next frame immediately */
    redraw();
}



HRESULT MxGlfwApplication::createContext(
        const MxSimulator::Config &conf)
{
    const Vector2 dpiScaling = this->dpiScaling({});
    Configuration c;
    c.setTitle(conf.title())
                .setSize(conf.size(), dpiScaling)
                .setWindowFlags(Configuration::WindowFlag::Resizable);
    GLConfiguration glConf;
    glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);

    bool b = tryCreate(c);

    if(!b) {
        return E_FAIL;
    }

    _win = new MxGlfwWindow(this->window());

    _ren = new MxUniverseRenderer{_win, 0.25};

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

MxUniverseRenderer* MxGlfwApplication::getRenderer()
{
    return _ren;
}

HRESULT MxGlfwApplication:: MxGlfwApplication::run()
{
    return exec();
}



HRESULT MxGlfwApplication::mainLoopIteration(double timeout) {
    GlfwApplication::mainLoopIteration();
    return S_OK;
}


