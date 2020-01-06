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


struct MxApplication : CObject
{
public:

    typedef MxApplicationConfig Configuration;


    static MxApplication *get();

    static HRESULT create(int argv, char** argc, const Configuration& conf);

    static HRESULT destroy();

    class MxApplicationImpl *impl;
};

class MxApplicationImpl {
public:

    virtual ~MxApplicationImpl() {};
};

/**
 * The type object for a MxSymbol.
 */
CAPI_DATA(PyTypeObject) *MxApplication_Type;

HRESULT MxApplication_init(PyObject *m);

PyObject *MxApplication_New(int argc, char** argv, const MxApplicationConfig *conf);

#endif /* SRC_MXAPPLICATION_H_ */
