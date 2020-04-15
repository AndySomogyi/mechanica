/*
 * MxGlfwApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <MxGlfwApplication.h>


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

#ifdef MAGNUM_TARGET_GL
#include "Magnum/GL/Version.h"
#include "Magnum/Platform/GLContext.h"
#endif



#ifdef GLFW_TRUE
/* The docs say that it's the same, verify that just in case */
static_assert(GLFW_TRUE == true && GLFW_FALSE == false, "GLFW does not have sane bool values");
#endif

enum class MxGlfwApplication::Flag: UnsignedByte {
    Redraw = 1 << 0,
    TextInputActive = 1 << 1,
    #ifdef CORRADE_TARGET_APPLE
    HiDpiWarningPrinted = 1 << 2
    #elif defined(CORRADE_TARGET_WINDOWS)
    /* On Windows, GLFW fires a viewport event already when creating the
       window, which means viewportEvent() gets called even before the
       constructor exits. That's not a problem if the window is created
       implicitly (because derived class vtable is not setup yet and so the
       call goes into the base class no-op viewportEvent()), but when calling
       create() / tryCreate() from user constructor, this might lead to crashes
       as things touched by viewportEvent() might not be initialized yet. To
       fix this, we ignore the first ever viewport event. This behavior was not
       observed on Linux or macOS (and thus ignoring the first viewport event
       there may be harmful), so keeping this Windows-only. */
    FirstViewportEventIgnored = 1 << 2
    #endif
};

MxGlfwApplication::MxGlfwApplication(const Arguments& arguments): MxGlfwApplication{arguments, Configuration{}} {}

MxGlfwApplication::MxGlfwApplication(const Arguments& arguments, const Configuration& configuration): MxGlfwApplication{arguments, NoCreate} {
    create(configuration);
}

#ifdef MAGNUM_TARGET_GL
MxGlfwApplication::MxGlfwApplication(const Arguments& arguments, const Configuration& configuration, const GLConfiguration& glConfiguration): MxGlfwApplication{arguments, NoCreate} {
    create(configuration, glConfiguration);
}
#endif

MxGlfwApplication::MxGlfwApplication(const Arguments& arguments, NoCreateT):
    _flags{Flag::Redraw}
{
    Utility::Arguments args{Magnum::Platform::Implementation::windowScalingArguments()};
    #ifdef MAGNUM_TARGET_GL
    _context.reset(new GLContext{NoCreate, args, arguments.argc, arguments.argv});
    #else
    /** @todo this is duplicated here and in Sdl2Application, figure out a nice
        non-duplicated way to handle this */
    args.addOption("log", "default").setHelp("log", "console logging", "default|quiet|verbose")
        .setFromEnvironment("log")
        .parse(arguments.argc, arguments.argv);
    #endif

    /* Init GLFW */
    glfwSetErrorCallback([](int, const char* const description) {
        Error{} << description;
    });

    if(!glfwInit()) {
        Error() << "Could not initialize GLFW";
        std::exit(8);
    }

    /* Save command-line arguments */
    if(args.value("log") == "verbose") _verboseLog = true;
    const std::string dpiScaling = args.value("dpi-scaling");
    if(dpiScaling == "default")
        _commandLineDpiScalingPolicy = Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Default;
    #ifdef CORRADE_TARGET_APPLE
    else if(dpiScaling == "framebuffer")
        _commandLineDpiScalingPolicy = Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Framebuffer;
    #else
    else if(dpiScaling == "virtual")
        _commandLineDpiScalingPolicy = Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Virtual;
    else if(dpiScaling == "physical")
        _commandLineDpiScalingPolicy = Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Physical;
    #endif
    else if(dpiScaling.find_first_of(" \t\n") != std::string::npos)
        _commandLineDpiScaling = args.value<Vector2>("dpi-scaling");
    else
        _commandLineDpiScaling = Vector2{args.value<Float>("dpi-scaling")};
}

void MxGlfwApplication::create() {
    create(Configuration{});
}

void MxGlfwApplication::create(const Configuration& configuration) {
    if(!tryCreate(configuration)) std::exit(1);
}

#ifdef MAGNUM_TARGET_GL
void MxGlfwApplication::create(const Configuration& configuration, const GLConfiguration& glConfiguration) {
    if(!tryCreate(configuration, glConfiguration)) std::exit(1);
}
#endif

Vector2 MxGlfwApplication::dpiScaling(const Configuration& configuration) {
    std::ostream* verbose = _verboseLog ? Debug::output() : nullptr;

    /* Print a helpful warning in case some extra steps are needed for HiDPI
       support */
    #ifdef CORRADE_TARGET_APPLE
    if(!Magnum::Platform::Implementation::isAppleBundleHiDpiEnabled() && !(_flags & Flag::HiDpiWarningPrinted)) {
        Warning{} << "Platform::MxGlfwApplication: warning: the executable is not a HiDPI-enabled app bundle";
        _flags |= Flag::HiDpiWarningPrinted;
    }
    #elif defined(CORRADE_TARGET_WINDOWS)
    /** @todo */
    #endif

    /* Use values from the configuration only if not overriden on command line
       to something non-default. In any case explicit scaling has a precedence
       before the policy. */
    Magnum::Platform::Implementation::GlfwDpiScalingPolicy dpiScalingPolicy{};
    if(!_commandLineDpiScaling.isZero()) {
        Debug{verbose} << "Platform::MxGlfwApplication: user-defined DPI scaling" << _commandLineDpiScaling.x();
        return _commandLineDpiScaling;
    } else if(_commandLineDpiScalingPolicy != Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Default) {
        dpiScalingPolicy = _commandLineDpiScalingPolicy;
    } else if(!configuration.dpiScaling().isZero()) {
        Debug{verbose} << "Platform::MxGlfwApplication: app-defined DPI scaling" << _commandLineDpiScaling.x();
        return configuration.dpiScaling();
    } else {
        dpiScalingPolicy = configuration.dpiScalingPolicy();
    }

    /* There's no choice on Apple, it's all controlled by the plist file. So
       unless someone specified custom scaling via config or command-line
       above, return the default. */
    #ifdef CORRADE_TARGET_APPLE
    return Vector2{1.0f};

    /* Otherwise there's a choice between virtual and physical DPI scaling */
    #else
    /* Try to get virtual DPI scaling first, if supported and requested */
    if(dpiScalingPolicy == Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Virtual) {
        /* Use Xft.dpi on X11. This could probably be dropped for GLFW 3.3+
           as glfwGetMonitorContentScale() does the same, but I'd still need to
           keep it for 2.2 and below, plus the same code needs to be used for
           SDL anyway. So keeping it to reduce the chance for unexpected minor
           differences across app implementations. */
        #ifdef _MAGNUM_PLATFORM_USE_X11
        const Vector2 dpiScaling{Magnum::Platform::Implementation::x11DpiScaling()};
        if(!dpiScaling.isZero()) {
            Debug{verbose} << "Platform::MxGlfwApplication: virtual DPI scaling" << dpiScaling.x();
            return dpiScaling;
        }

        /* Check for DPI awareness on non-RT Windows and then ask for content
           scale (available since GLFW 3.3). GLFW is advertising the
           application to be DPI-aware on its own even without supplying an
           explicit manifest -- https://github.com/glfw/glfw/blob/089ea9af227fdffdf872348923e1c12682e63029/src/win32_init.c#L564-L569
           If, for some reason, the app is still not DPI-aware, tell that to
           the user explicitly and don't even attempt to query the value if the
           app is not DPI aware. If it's desired to get the DPI value
           unconditionally, the user should use physical DPI scaling instead. */
        #elif defined(CORRADE_TARGET_WINDOWS) && !defined(CORRADE_TARGET_WINDOWS_RT)
        if(!Magnum::Platform::Implementation::isWindowsAppDpiAware()) {
            Warning{verbose} << "Platform::MxGlfwApplication: your application is not set as DPI-aware, DPI scaling won't be used";
            return Vector2{1.0f};
        }
        #if GLFW_VERSION_MAJOR*100 + GLFW_VERSION_MINOR >= 303
        GLFWmonitor* const monitor = glfwGetPrimaryMonitor();
        Vector2 dpiScaling;
        glfwGetMonitorContentScale(monitor, &dpiScaling.x(), &dpiScaling.y());
        Debug{verbose} << "Platform::MxGlfwApplication: virtual DPI scaling" << dpiScaling;
        return dpiScaling;
        #else
        Debug{verbose} << "Platform::MxGlfwApplication: sorry, virtual DPI scaling only available on GLFW 3.3+, falling back to physical DPI scaling";
        #endif

        /* Otherwise ¯\_(ツ)_/¯ */
        #else
        Debug{verbose} << "Platform::MxGlfwApplication: sorry, virtual DPI scaling not implemented on this platform yet, falling back to physical DPI scaling";
        #endif
    }

    /* At this point, either the virtual DPI query failed or a physical DPI
       scaling is requested */
    CORRADE_INTERNAL_ASSERT(dpiScalingPolicy == Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Virtual || dpiScalingPolicy == Magnum::Platform::Implementation::GlfwDpiScalingPolicy::Physical);

    /* Physical DPI scaling. Enable only on Linux (where it gets the usually
       very-off value from X11) and on non-RT Windows (where it calculates it
       from actual monitor dimensions). */
    #if defined(CORRADE_TARGET_UNIX) || (defined(CORRADE_TARGET_WINDOWS) && !defined(CORRADE_TARGET_WINDOWS_RT))
    GLFWmonitor* const monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* const mode = glfwGetVideoMode(monitor);
    Vector2i monitorSize;
    glfwGetMonitorPhysicalSize(monitor, &monitorSize.x(), &monitorSize.y());
    if(monitorSize.isZero()) {
        Warning{verbose} << "Platform::MxGlfwApplication: the physical monitor size is zero? DPI scaling won't be used";
        return Vector2{1.0f};
    }
    auto dpi = Vector2{Vector2i{mode->width, mode->height}*25.4f/Vector2{monitorSize}};
    const Vector2 dpiScaling{dpi/96.0f};
    Debug{verbose} << "Platform::MxGlfwApplication: physical DPI scaling" << dpiScaling;
    return dpiScaling;

    /* Not implemented otherwise */
    #else
    Debug{verbose} << "Platform::MxGlfwApplication: sorry, physical DPI scaling not implemented on this platform yet";
    return Vector2{1.0f};
    #endif
    #endif
}

bool MxGlfwApplication::tryCreate(const Configuration& configuration) {
    #ifdef MAGNUM_TARGET_GL
    #ifdef GLFW_NO_API
    if(!(configuration.windowFlags() & Configuration::WindowFlag::Contextless))
    #endif
    {
        return tryCreate(configuration, GLConfiguration{});
    }
    #endif

    /* Scale window based on DPI */
    _dpiScaling = dpiScaling(configuration);
    const Vector2i scaledWindowSize = configuration.size()*_dpiScaling;

    /* Window flags */
    GLFWmonitor* monitor = nullptr; /* Needed for setting fullscreen */
    if (configuration.windowFlags() >= Configuration::WindowFlag::Fullscreen) {
        monitor = glfwGetPrimaryMonitor();
        glfwWindowHint(GLFW_AUTO_ICONIFY, configuration.windowFlags() >= Configuration::WindowFlag::AutoIconify);
    } else {
        const Configuration::WindowFlags& flags = configuration.windowFlags();
        glfwWindowHint(GLFW_RESIZABLE, flags >= Configuration::WindowFlag::Resizable);
        glfwWindowHint(GLFW_VISIBLE, !(flags >= Configuration::WindowFlag::Hidden));
        #ifdef GLFW_MAXIMIZED
        glfwWindowHint(GLFW_MAXIMIZED, flags >= Configuration::WindowFlag::Maximized);
        #endif
        glfwWindowHint(GLFW_FLOATING, flags >= Configuration::WindowFlag::Floating);
    }
    glfwWindowHint(GLFW_FOCUSED, configuration.windowFlags() >= Configuration::WindowFlag::Focused);

    #ifdef GLFW_NO_API
    /* Disable implicit GL context creation */
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    #endif



    return true;
}



#ifdef MAGNUM_TARGET_GL
bool MxGlfwApplication::tryCreate(const Configuration& configuration, const GLConfiguration& glConfiguration) {
    CORRADE_ASSERT( _context->version() == GL::Version::None, "Platform::GlfwApplication::tryCreate(): window with OpenGL context already created", false);

    /* Scale window based on DPI */
    _dpiScaling = dpiScaling(configuration);
    const Vector2i scaledWindowSize = configuration.size()*_dpiScaling;

    /* Window flags */
    GLFWmonitor* monitor = nullptr; /* Needed for setting fullscreen */
    if (configuration.windowFlags() >= Configuration::WindowFlag::Fullscreen) {
        monitor = glfwGetPrimaryMonitor();
        glfwWindowHint(GLFW_AUTO_ICONIFY, configuration.windowFlags() >= Configuration::WindowFlag::AutoIconify);
    } else {
        const Configuration::WindowFlags& flags = configuration.windowFlags();
        glfwWindowHint(GLFW_RESIZABLE, flags >= Configuration::WindowFlag::Resizable);
        glfwWindowHint(GLFW_VISIBLE, !(flags >= Configuration::WindowFlag::Hidden));
        #ifdef GLFW_MAXIMIZED
        glfwWindowHint(GLFW_MAXIMIZED, flags >= Configuration::WindowFlag::Maximized);
        #endif
        glfwWindowHint(GLFW_FLOATING, flags >= Configuration::WindowFlag::Floating);
    }
    glfwWindowHint(GLFW_FOCUSED, configuration.windowFlags() >= Configuration::WindowFlag::Focused);

    /* Framebuffer setup */
    glfwWindowHint(GLFW_RED_BITS, glConfiguration.colorBufferSize().r());
    glfwWindowHint(GLFW_GREEN_BITS, glConfiguration.colorBufferSize().g());
    glfwWindowHint(GLFW_BLUE_BITS, glConfiguration.colorBufferSize().b());
    glfwWindowHint(GLFW_ALPHA_BITS, glConfiguration.colorBufferSize().a());
    glfwWindowHint(GLFW_DEPTH_BITS, glConfiguration.depthBufferSize());
    glfwWindowHint(GLFW_STENCIL_BITS, glConfiguration.stencilBufferSize());
    glfwWindowHint(GLFW_SAMPLES, glConfiguration.sampleCount());
    glfwWindowHint(GLFW_SRGB_CAPABLE, glConfiguration.isSrgbCapable());

    /* Request debug context if --magnum-gpu-validation is enabled */
    GLConfiguration::Flags glFlags = glConfiguration.flags();
    if(_context->internalFlags() & GL::Context::InternalFlag::GpuValidation)
        glFlags |= GLConfiguration::Flag::Debug;

    #ifdef GLFW_CONTEXT_NO_ERROR
    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, glFlags >= GLConfiguration::Flag::NoError);
    #endif
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, glFlags >= GLConfiguration::Flag::Debug);
    glfwWindowHint(GLFW_STEREO, glFlags >= GLConfiguration::Flag::Stereo);

    /* Set context version, if requested */
    if(glConfiguration.version() != GL::Version::None) {
        Int major, minor;
        std::tie(major, minor) = version(glConfiguration.version());
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
        #ifndef MAGNUM_TARGET_GLES
        if(glConfiguration.version() >= GL::Version::GL320) {
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, glFlags >= GLConfiguration::Flag::ForwardCompatible);
        }
        #else
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
        #endif

    /* Request usable version otherwise */
    } else {
        #ifndef MAGNUM_TARGET_GLES
        /* First try to create core context. This is needed mainly on macOS and
           Mesa, as support for recent OpenGL versions isn't implemented in
           compatibility contexts (which are the default). Unlike SDL2, GLFW
           requires at least version 3.2 to be able to request a core profile. */
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, glFlags >= GLConfiguration::Flag::ForwardCompatible);
        #else
        /* For ES the major context version is compile-time constant */
        #ifdef MAGNUM_TARGET_GLES3
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        #elif defined(MAGNUM_TARGET_GLES2)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        #else
        #error unsupported OpenGL ES version
        #endif
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
        #endif
    }

    /* Create window. Hide it by default so we don't have distracting window
       blinking in case we have to destroy it again right away. If the creation
       succeeds, make the context current so we can query GL_VENDOR below.
       If we are on Wayland, this is causing a segfault; a blinking window is
       acceptable in this case. */
    constexpr const char waylandString[] = "wayland";
    const char* const xdgSessionType = std::getenv("XDG_SESSION_TYPE");
    if(!xdgSessionType || std::strncmp(xdgSessionType, waylandString, sizeof(waylandString)) != 0)
        glfwWindowHint(GLFW_VISIBLE, false);
    else if(_verboseLog)
        Warning{} << "Platform::GlfwApplication: Wayland detected, GL context has to be created with the window visible and may cause flicker on startup";


    #ifndef MAGNUM_TARGET_GLES
    /* Fall back to (forward compatible) GL 2.1, if version is not
       user-specified and either core context creation fails or we are on
       binary NVidia/AMD drivers on Linux/Windows or Intel Windows drivers.
       Instead of creating forward-compatible context with highest available
       version, they force the version to the one specified, which is
       completely useless behavior. */
    #ifndef CORRADE_TARGET_APPLE
    constexpr static const char nvidiaVendorString[] = "NVIDIA Corporation";
    #ifdef CORRADE_TARGET_WINDOWS
    constexpr static const char intelVendorString[] = "Intel";
    #endif
    constexpr static const char amdVendorString[] = "ATI Technologies Inc.";
    const char* vendorString;
    #endif
    if(glConfiguration.version() == GL::Version::None
        #ifndef CORRADE_TARGET_APPLE
        /* If context creation fails *really bad*, glGetString() may actually
           return nullptr. Check for that to avoid crashes deep inside
           strncmp(). Sorry about the UGLY code, HOPEFULLY THERE WON'T BE MORE
           WORKAROUNDS */
        || (vendorString = reinterpret_cast<const char*>(glGetString(GL_VENDOR)),
        vendorString && (std::strncmp(vendorString, nvidiaVendorString, sizeof(nvidiaVendorString)) == 0 ||
         #ifdef CORRADE_TARGET_WINDOWS
         std::strncmp(vendorString, intelVendorString, sizeof(intelVendorString)) == 0 ||
         #endif
         std::strncmp(vendorString, amdVendorString, sizeof(amdVendorString)) == 0)
         && !_context->isDriverWorkaroundDisabled("no-forward-compatible-core-context"))
         #endif
    ) {


        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE);
        /** @todo or keep the fwcompat? */
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, false);
    }
    #endif




    /* Destroy everything when the Magnum context creation fails */
    if(!_context->tryCreate()) {
        return false;
    }

    /* Return true if the initialization succeeds */
    return true;
}
#endif

