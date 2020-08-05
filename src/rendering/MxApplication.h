/*
 * MxApplication.h
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#ifndef SRC_MXAPPLICATION_H_
#define SRC_MXAPPLICATION_H_

#include <Mechanica.h>
#include "mechanica_private.h"

#include <Magnum/GL/Context.h>

#include <GLFW/glfw3.h>

#include <rendering/MxUniverseRenderer.h>
#include <rendering/MxGlfwWindow.h>



enum MxWindowAttributes {
    MX_FOCUSED = GLFW_FOCUSED,
    MX_ICONIFIED = GLFW_ICONIFIED,
    MX_RESIZABLE = GLFW_RESIZABLE,
    MX_VISIBLE = GLFW_VISIBLE,
    MX_DECORATED = GLFW_DECORATED,
    MX_AUTO_ICONIFY = GLFW_AUTO_ICONIFY,
    MX_FLOATING = GLFW_FLOATING,
    MX_MAXIMIZED = GLFW_MAXIMIZED,
    MX_CENTER_CURSOR = GLFW_CENTER_CURSOR,
    MX_TRANSPARENT_FRAMEBUFFER = GLFW_TRANSPARENT_FRAMEBUFFER,
    MX_HOVERED = GLFW_HOVERED,
    MX_FOCUS_ON_SHOW = GLFW_FOCUS_ON_SHOW,
    MX_RED_BITS = GLFW_RED_BITS,
    MX_GREEN_BITS = GLFW_GREEN_BITS,
    MX_BLUE_BITS = GLFW_BLUE_BITS,
    MX_ALPHA_BITS = GLFW_ALPHA_BITS,
    MX_DEPTH_BITS = GLFW_DEPTH_BITS,
    MX_STENCIL_BITS = GLFW_STENCIL_BITS,
    MX_ACCUM_RED_BITS = GLFW_ACCUM_RED_BITS,
    MX_ACCUM_GREEN_BITS = GLFW_ACCUM_GREEN_BITS,
    MX_ACCUM_BLUE_BITS = GLFW_ACCUM_BLUE_BITS,
    MX_ACCUM_ALPHA_BITS = GLFW_ACCUM_ALPHA_BITS,
    MX_AUX_BUFFERS = GLFW_AUX_BUFFERS,
    MX_STEREO = GLFW_STEREO,
    MX_SAMPLES = GLFW_SAMPLES,
    MX_SRGB_CAPABLE = GLFW_SRGB_CAPABLE,
    MX_REFRESH_RATE = GLFW_REFRESH_RATE,
    MX_DOUBLEBUFFER = GLFW_DOUBLEBUFFER,
    MX_CLIENT_API = GLFW_CLIENT_API,
    MX_CONTEXT_VERSION_MAJOR = GLFW_CONTEXT_VERSION_MAJOR,
    MX_CONTEXT_VERSION_MINOR = GLFW_CONTEXT_VERSION_MINOR,
    MX_CONTEXT_REVISION = GLFW_CONTEXT_REVISION,
    MX_CONTEXT_ROBUSTNESS = GLFW_CONTEXT_ROBUSTNESS,
    MX_OPENGL_FORWARD_COMPAT = GLFW_OPENGL_FORWARD_COMPAT,
    MX_OPENGL_DEBUG_CONTEXT = GLFW_OPENGL_DEBUG_CONTEXT,
    MX_OPENGL_PROFILE = GLFW_OPENGL_PROFILE,
    MX_CONTEXT_RELEASE_BEHAVIOR = GLFW_CONTEXT_RELEASE_BEHAVIOR,
    MX_CONTEXT_NO_ERROR = GLFW_CONTEXT_NO_ERROR,
    MX_CONTEXT_CREATION_API = GLFW_CONTEXT_CREATION_API,
    MX_SCALE_TO_MONITOR = GLFW_SCALE_TO_MONITOR,
    MX_COCOA_RETINA_FRAMEBUFFER = GLFW_COCOA_RETINA_FRAMEBUFFER,
    MX_COCOA_FRAME_NAME = GLFW_COCOA_FRAME_NAME,
    MX_COCOA_GRAPHICS_SWITCHING = GLFW_COCOA_GRAPHICS_SWITCHING,
    MX_X11_CLASS_NAME = GLFW_X11_CLASS_NAME,
    MX_X11_INSTANCE_NAME = GLFW_X11_INSTANCE_NAME
};




/**
 * Set config options for opengl for now.
 */
struct MxApplicationConfig {
public:
    /**
     * @brief Window flag
     *
     * @see @ref WindowFlags, @ref setWindowFlags()
     */
    enum WindowFlag  {
        None = 0,
        Fullscreen = 1 << 0,   /**< Fullscreen window */
        Resizable = 1 << 1,    /**< Resizable window */
        Hidden = 1 << 2,       /**< Hidden window */


        Maximized = 1 << 3,


        Minimized = 1 << 4,    /**< Minimized window */
        Floating = 1 << 5,     /**< Window floating above others, top-most */

        /**
         * Automatically iconify (minimize) if fullscreen window loses
         * input focus
         */
        AutoIconify = 1 << 6,

        Focused = 1 << 7,      /**< Window has input focus */


        /**
         * Do not create any GPU context. Use together with
         * @ref GlfwApplication(const Arguments&),
         * @ref GlfwApplication(const Arguments&, const Configuration&),
         * @ref create(const Configuration&) or
         * @ref tryCreate(const Configuration&) to prevent implicit
         * creation of an OpenGL context.
         *
         * @note Supported since GLFW 3.2.
         */
        Contextless = 1 << 8
    };

    unsigned windowFlag = 0;


    /**
     * @brief DPI scaling policy
     *
     * DPI scaling policy when requesting a particular window size. Can
     * be overriden on command-line using `--magnum-dpi-scaling` or via
     * the `MAGNUM_DPI_SCALING` environment variable.
     * @see @ref setSize(), @ref Platform-Sdl2Application-dpi
     */
    enum class DpiScalingPolicy {
        /**
         * Framebuffer DPI scaling. The window will have the same size as
         * requested, but the framebuffer size will be different. Supported
         * only on macOS and iOS and is also the only supported value
         * there.
         */
        Framebuffer,

        /**
         * Virtual DPI scaling. Scales the window based on UI scaling
         * setting in the system. Falls back to
         * @ref DpiScalingPolicy::Physical on platforms that don't support
         * it. Supported only on desktop platforms (except macOS) and it's
         * the default there.
         *
         * Equivalent to `--magnum-dpi-scaling virtual` passed on
         * command-line.
         */
        Virtual,

        /**
         * Physical DPI scaling. Takes the requested window size as a
         * physical size that a window would have on platform's default DPI
         * and scales it to have the same size on given display physical
         * DPI. On platforms that don't have a concept of a window it
         * causes the framebuffer to match screen pixels 1:1 without any
         * scaling. Supported on desktop platforms except macOS and on
         * mobile and web. Default on mobile and web.
         *
         * Equivalent to `--magnum-dpi-scaling physical` passed on
         * command-line.
         */
        Physical,

        /**
         * Default policy for current platform. Alias to one of
         * @ref DpiScalingPolicy::Framebuffer, @ref DpiScalingPolicy::Virtual
         * or @ref DpiScalingPolicy::Physical depending on platform. See
         * @ref Platform-Sdl2Application-dpi for details.
         */
        Default
    };
};


struct MxApplication
{
public:

    /**
     * python list of windows.
     *
     * We do some pretty low level stuff with window events, so keep then as python objects.
     */
    PyObject *windows;

    virtual ~MxApplication() {};


    /**
     * This function processes only those events that are already in the event
     * queue and then returns immediately. Processing events will cause the window
     * and input callbacks associated with those events to be called.
     *
     * On some platforms, a window move, resize or menu operation will cause
     * event processing to block. This is due to how event processing is designed
     * on those platforms. You can use the window refresh callback to redraw the
     * contents of your window when necessary during such operations.
     */
    virtual HRESULT pollEvents () = 0;

    /**
     *   This function puts the calling thread to sleep until at least one
     *   event is available in the event queue. Once one or more events are
     *   available, it behaves exactly like glfwPollEvents, i.e. the events
     *   in the queue are processed and the function then returns immediately.
     *   Processing events will cause the window and input callbacks associated
     *   with those events to be called.
     *
     *   Since not all events are associated with callbacks, this function may return
     *   without a callback having been called even if you are monitoring all callbacks.
     *
     *  On some platforms, a window move, resize or menu operation will cause event
     *  processing to block. This is due to how event processing is designed on
     *  those platforms. You can use the window refresh callback to redraw the
     *  contents of your window when necessary during such operations.
     */
    virtual HRESULT waitEvents () = 0;

    /**
     * This function puts the calling thread to sleep until at least
     * one event is available in the event queue, or until the specified
     * timeout is reached. If one or more events are available, it behaves
     * exactly like pollEvents, i.e. the events in the queue are
     * processed and the function then returns immediately. Processing
     * events will cause the window and input callbacks associated with those
     * events to be called.
     *
     * The timeout value must be a positive finite number.
     * Since not all events are associated with callbacks, this function may
     * return without a callback having been called even if you are monitoring
     * all callbacks.
     *
     * On some platforms, a window move, resize or menu operation will cause
     * event processing to block. This is due to how event processing is designed
     * on those platforms. You can use the window refresh callback to redraw the
     * contents of your window when necessary during such operations.
     */

    virtual HRESULT waitEventsTimeout(double  timeout) = 0;


    /**
     * This function posts an empty event from the current thread
     * to the event queue, causing waitEvents or waitEventsTimeout to return.
     */
    virtual HRESULT postEmptyEvent() = 0;


    virtual HRESULT mainLoopIteration(double timeout) { return E_NOTIMPL; };


    virtual HRESULT setSwapInterval(int si) = 0;


    // temporary hack until we setup events correctly
    virtual MxGlfwWindow *getWindow() {
        return NULL;
    }

    virtual int windowAttribute(MxWindowAttributes attr) { return E_NOTIMPL;};

    virtual HRESULT setWindowAttribute(MxWindowAttributes attr, int val) { return E_NOTIMPL;};


    virtual MxUniverseRenderer *getRenderer() {
        return NULL;
    }
    
    /**
     * post a re-draw event, to tell the renderer
     * that it should re-draw
     */
    virtual HRESULT redraw() {
        return E_NOTIMPL;
    }

    virtual HRESULT run() { return E_NOTIMPL;};


    // soft hide the window
    virtual HRESULT close() { return E_NOTIMPL; };

    // hard window close
    virtual HRESULT destroy() { return E_NOTIMPL; };

    // display the window if closed.
    virtual HRESULT show() { return E_NOTIMPL; };

};




#endif /* SRC_MXAPPLICATION_H_ */
