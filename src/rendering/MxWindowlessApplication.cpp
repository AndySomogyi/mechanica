/*
 * MxWindowlessApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <rendering/MxWindowlessApplication.h>
#include <rendering/MxWindow.h>
#include <rendering/MxUniverseRenderer.h>

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>

#include <iostream>

#include "access_private.hpp"



namespace {
    namespace private_access_detail {
      /* Tag type, used to declare different get funcitons for different
       * members
       */
      struct PrivateAccessTag8 {};
      /* Explicit instantiation */
      template struct private_access<__decltype(&Magnum::Platform::WindowlessApplication::_context), &Magnum::Platform::WindowlessApplication::_context,
                                     PrivateAccessTag8>;
      /* We can build the PtrType only with two aliases */
      /* E.g. using PtrType = int(int) *; would be illformed */
      using Alias_PrivateAccessTag8 = Containers::Pointer<Platform::GLContext>;
      using PtrType_PrivateAccessTag8 =
          Alias_PrivateAccessTag8 Magnum::Platform::WindowlessApplication::*;
      /* Declare the friend function, now it is visible in namespace scope.
       * Note,
       * we could declare it inside the Tag type too, in that case ADL would
       * find
       * the declaration. By choosing to declare it here, the Tag type remains
       * a
       * simple tag type, it has no other responsibilities. */
      PtrType_PrivateAccessTag8 get(PrivateAccessTag8);
    }
  }
  namespace {
    namespace access_private {
      Containers::Pointer<Platform::GLContext> &_context(Magnum::Platform::WindowlessApplication &&t) { return t.*get(private_access_detail::PrivateAccessTag8{}); }
      Containers::Pointer<Platform::GLContext> &_context(Magnum::Platform::WindowlessApplication &t) { return t.*get(private_access_detail::PrivateAccessTag8{}); }
      /* The following usings are here to avoid duplicate const qualifier
       * warnings
       */
      using XPrivateAccessTag8 = Containers::Pointer<Platform::GLContext>;
      using YPrivateAccessTag8 =
          const XPrivateAccessTag8;
      YPrivateAccessTag8 & _context(const Magnum::Platform::WindowlessApplication &t) {
        return t.*get(private_access_detail::PrivateAccessTag8{});
      }
    }
  }


ACCESS_PRIVATE_FIELD(Magnum::Platform::WindowlessApplication, Magnum::Platform::WindowlessGLContext, _glContext);



/**
 *
 */
struct MxWindowlessWindow : MxWindow
{
    MxWindowlessApplication *app;

    Magnum::Vector2i windowSize() const override {
        return app->framebuffer().viewport().size();
    };

    void redraw() override {
        app->redraw();
    }

    Magnum::GL::AbstractFramebuffer &framebuffer() override {
        return app->frameBuffer;
    }

    /**
     * attach to an existing GLFW Window
     */
    MxWindowlessWindow(MxWindowlessApplication *_app) : app{_app} {
    };
};

MxWindowlessApplication::~MxWindowlessApplication()
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}

MxWindowlessApplication::MxWindowlessApplication(const Arguments &args) :
    WindowlessApplication{args, Magnum::NoCreate},
    renderBuffer{Magnum::NoCreate},
    frameBuffer{Magnum::NoCreate},
    depthStencil{Magnum::NoCreate}
{
}

HRESULT MxWindowlessApplication::createContext(const MxSimulator::Config &conf) {

    // default Magnum WindowlessApplication config, does not have any options
    Configuration windowlessConf;

    Log(LOG_INFORMATION) << "trying to create windowless context";

    if(!WindowlessApplication::tryCreateContext(windowlessConf)) {
        return c_error(E_FAIL, "could not create windowless context");
    }
    
    Magnum::Platform::WindowlessApplication &app = *this;
    Magnum::Platform::WindowlessGLContext &glContext = access_private::_glContext(app);
    Containers::Pointer<Platform::GLContext> &context = access_private::_context(app);
    
    
    
#if defined(MX_APPLE)
    const char* cname = "CGL Context";
#elif defined(MX_LINUX)
    const char* cname = "EGL Context";
#elif defined(MX_WINDOWS)
    const char* cname = "WGL Context";
#else
#error "NO GL Supported"
#endif


    Log(LOG_NOTICE) << "created windowless context, " << cname << glContext.glContext();
    
    Log(LOG_NOTICE) << "GL Version: " << context->versionString();
    Log(LOG_NOTICE) << "GL Vendor: " << context->vendorString();
    Log(LOG_NOTICE) << "GL Renderer: " << context->rendererString();
    
    for(auto s : context->extensionStrings()) {
        Log(LOG_NOTICE) << "GL Extension: " << s;
    }
    
    Vector2i size = conf.windowSize();

    // create the render buffers here, after we have a context,
    // default ctor makes this with a {Magnum::NoCreate},
    renderBuffer = Magnum::GL::Renderbuffer();
    depthStencil = Magnum::GL::Renderbuffer();

    depthStencil.setStorage(GL::RenderbufferFormat::Depth24Stencil8, size);

    renderBuffer.setStorage(Magnum::GL::RenderbufferFormat::RGBA8, size);

    frameBuffer = Magnum::GL::Framebuffer{{{0,0}, size}};

    frameBuffer
        .attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderBuffer)
        .attachRenderbuffer(GL::Framebuffer::BufferAttachment::DepthStencil, depthStencil)
        .clear(GL::FramebufferClear::Color)
        .bind();

    window = new MxWindowlessWindow(this);

    // renderer accesses the framebuffer from the window handle we pass in.
    renderer = new MxUniverseRenderer(conf, window);

    return S_OK;
}


MxUniverseRenderer *MxWindowlessApplication::getRenderer() {
    return renderer;
}

HRESULT MxWindowlessApplication::MxWindowlessApplication::pollEvents()
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::MxWindowlessApplication::waitEvents()
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::MxWindowlessApplication::waitEventsTimeout(
        double timeout)
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::MxWindowlessApplication::postEmptyEvent()
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::mainLoopIteration(double timeout)
{
    return E_NOTIMPL;
}

struct MxGlfwWindow* MxWindowlessApplication::getWindow()
{
    return NULL;
}

int MxWindowlessApplication::windowAttribute(MxWindowAttributes attr)
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::setWindowAttribute(MxWindowAttributes attr,
        int val)
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::redraw()
{
    Log(LOG_TRACE);
    
    // TODO: need to re-evaluate drawing, should not have to check...
    // drawing code on the wrong thread should not call re-draw, only the renderer should
    
    if(Magnum::GL::Context::hasCurrent()) {
        //frameBuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderBuffer)
        frameBuffer
            .clear(GL::FramebufferClear::Color)
            .bind();

        /* Draw particles */
        renderer->draw();
    }

    return S_OK;
}

HRESULT MxWindowlessApplication::close()
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::destroy()
{
    return E_NOTIMPL;
}

HRESULT MxWindowlessApplication::show()
{
    return redraw();
}

HRESULT MxWindowlessApplication::messageLoop(double et)
{
    return E_NOTIMPL;
}

Magnum::GL::AbstractFramebuffer& MxWindowlessApplication::framebuffer() {
    return frameBuffer;
}

bool MxWindowlessApplication::contextMakeCurrent()
{
    Log(LOG_TRACE);
    
    Magnum::Platform::WindowlessApplication &app = *this;

    Magnum::Platform::WindowlessGLContext &glContext = access_private::_glContext(app);

    Containers::Pointer<Platform::GLContext> &context = access_private::_context(app);

    if(glContext.makeCurrent()) {
        Platform::GLContext *p = context.get();
        Magnum::GL::Context::makeCurrent(p);
        return true;
    }

    return false;
}

bool MxWindowlessApplication::contextHasCurrent()
{
    Log(LOG_TRACE);
    
    return Magnum::GL::Context::hasCurrent();
}

bool MxWindowlessApplication::contextRelease()
{
    Log(LOG_TRACE);
    
    Magnum::Platform::WindowlessApplication &app = *this;

    Magnum::Platform::WindowlessGLContext &context = access_private::_glContext(app);

    Magnum::GL::Context::makeCurrent(nullptr);

    return context.release();
}
