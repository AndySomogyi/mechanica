/*
 * MxSimulator.h
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#ifndef SRC_MXSIMULATOR_H_
#define SRC_MXSIMULATOR_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include "MxView.h"

#include "Magnum/Platform/GLContext.h"
#include "Magnum/Platform/Implementation/DpiScaling.h"
#include "MxUniverse.h"



enum MxSimulator_Key {
    MXSIMULATOR_NONE,
    MXSIMULATOR_WINDOWLESS,
    MXSIMULATOR_GLFW
};

struct MxSimulator_ConfigurationItem {
    uint32_t key;
    union {
        int intVal;
        int intVecVal[4];
    };
};

enum MxSimulator_Options {
    Windowless = 1 << 0,

    GLFW = 1 << 1,
    /**
     * Forward compatible context
     *
     * @requires_gl Core/compatibility profile distinction and forward
     *      compatibility applies only to desktop GL.
     */
    GlForwardCompatible = 1 << 2,

    /**
     * Specifies whether errors should be generated by the context.
     * If enabled, situations that would have generated errors instead
     * cause undefined behavior.
     *
     * @note Supported since GLFW 3.2.
     */
    GlNoError = 1 << 3,


    /**
     * Debug context. Enabled automatically if the
     * `--magnum-gpu-validation` @ref GL-Context-command-line "command-line option"
     * is present.
     */
    GlDebug = 1 << 4,

    GlStereo = 1 << 5     /**< Stereo rendering */
};




CAPI_DATA(PyTypeObject) MxSimulator_Type;

struct CAPI_EXPORT MxSimulator {

    class CAPI_EXPORT GLConfig;

    enum class DpiScalingPolicy : UnsignedByte {
        /* Using 0 for an "unset" value */

        #ifdef CORRADE_TARGET_APPLE
        Framebuffer = 1,
        #endif

        #ifndef CORRADE_TARGET_APPLE
        Virtual = 2,

        Physical = 3,
        #endif

        Default
            #ifdef CORRADE_TARGET_APPLE
            = Framebuffer
            #else
            = Virtual
            #endif
    };

    /**
     * @brief Window flag
     *
     * @see @ref WindowFlags, @ref setWindowFlags()
     */
    enum WindowFlags : UnsignedShort
    {
        /**< Fullscreen window */
        Fullscreen = 1 << 0,

        /**
         * No window decoration
         */
        Borderless = 1 << 1,

        Resizable = 1 << 2,    /**< Resizable window */
        Hidden = 1 << 3,       /**< Hidden window */


        /**
         * Maximized window
         *
         * @note Supported since GLFW 3.2.
         */
        Maximized = 1 << 4,


        Minimized = 1 << 5,    /**< Minimized window */

        /**
         * Always on top
         * @m_since_latest
         */
        AlwaysOnTop = 1 << 6,



        /**
         * Automatically iconify (minimize) if fullscreen window loses
         * input focus
         */
        AutoIconify = 1 << 7,

        /**
         * Window has input focus
         *
         * @todo there's also GLFW_FOCUS_ON_SHOW, what's the difference?
         */
        Focused = 1 << 8,

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
        Contextless = 1 << 9

    };



    struct CAPI_EXPORT Config
    {
    public:

        /**
         * @brief DPI scaling policy
         *
         * DPI scaling policy when requesting a particular window size. Can
         * be overriden on command-line using `--magnum-dpi-scaling` or via
         * the `MAGNUM_DPI_SCALING` environment variable.
         * @see @ref setSize(), @ref Platform-Sdl2Application-dpi
         */


        /*implicit*/
        Config();

        ~Config() {};

        /** @brief Window title */
        std::string title() const
        {
            return _title;
        }

        /**
         * @brief Set window title
         * @return Reference to self (for method chaining)
         *
         * Default is @cpp "Magnum GLFW Application" @ce.
         */
        void setTitle(std::string title)
        {
            _title = std::move(title);
        }

        /** @brief Window size */
        Vector2i windowSize() const
        {
            return _size;
        }

        /**
         * @brief DPI scaling policy
         *
         * If @ref dpiScaling() is non-zero, it has a priority over this value.
         * The `--magnum-dpi-scaling` command-line option has a priority over
         * any application-set value.
         * @see @ref setSize(const Vector2i&, DpiScalingPolicy)
         */
        DpiScalingPolicy dpiScalingPolicy() const
        {
            return _dpiScalingPolicy;
        }

        /**
         * @brief Custom DPI scaling
         *
         * If zero, then @ref dpiScalingPolicy() has a priority over this
         * value. The `--magnum-dpi-scaling` command-line option has a priority
         * over any application-set value.
         * @see @ref setSize(const Vector2i&, const Vector2&)
         * @todo change this on a DPI change event (GLFW 3.3 has a callback:
         *  https://github.com/mosra/magnum/issues/243#issuecomment-388384089)
         */
        Vector2 dpiScaling() const
        {
            return _dpiScaling;
        }

        void setDpiScaling(const Vector2 &vec)
           {
               _dpiScaling = vec;
           }


        void setSizeAndScaling(const Vector2i& size, DpiScalingPolicy dpiScalingPolicy = DpiScalingPolicy::Default) {
                    _size = size;
                    _dpiScalingPolicy = dpiScalingPolicy;

                }


        void setSizeAndScaling(const Vector2i& size, const Vector2& dpiScaling) {
                    _size = size;
                    _dpiScaling = dpiScaling;
        }

        /**
         * @brief Set window size
         * @param size              Desired window size
         * @param dpiScalingPolicy  Policy based on which DPI scaling will be set
         * @return Reference to self (for method chaining)
         *
         * Default is @cpp {800, 600} @ce. See @ref Platform-MxGlfwApplication-dpi
         * for more information.
         * @see @ref setSize(const Vector2i&, const Vector2&)
         */
        void setWindowSize(const Vector2i &size)
        {
            _size = size;
        }

        /** @brief Window flags */
        uint32_t windowFlags() const
        {
            return _windowFlags;
        }

        /**
         * @brief Set window flags
         * @return  Reference to self (for method chaining)
         *
         * Default is @ref WindowFlag::Focused.
         */
        void setWindowFlags(uint32_t windowFlags)
        {
            _windowFlags = windowFlags;
        }

        bool windowless() const {
            return _windowless;
        }

        void setWindowless(bool val) {
            _windowless = val;
        }

        int size() const {
            return universeConfig.nParticles;
        }

        void setSize(int i ) {
            universeConfig.nParticles = i;
        }

        MxUniverseConfig universeConfig;

        int queues;

        int argc = 0;

        char** argv = NULL;
        
        
        std::vector<Magnum::Vector4> clipPlanes;

    private:
        std::string _title;
        Vector2i _size;
        uint32_t _windowFlags;
        DpiScalingPolicy _dpiScalingPolicy;
        Vector2 _dpiScaling;
        bool _windowless;
    };
    
    struct MxUniverseRenderer *getRenderer();


    int32_t kind;
    struct MxApplication *app;


    // python list of windows.
    PyObject *windows;


    enum Flags {
        Running = 1 << 0
    };

    /**
     * gets the global simulator object, throws exception if fail.
     */
    static MxSimulator *Get();
};


CAPI_FUNC(HRESULT) MxSimulator_InitConfig(const MxSimulator::Config &conf,
        const MxSimulator::GLConfig &glConf);


/**
 * The global simulator object
 */
// CAPI_DATA(MxSimulator*) Simulator;

/**
 * Creates a new simulator if the global one does not exist,
 * returns the global if it does.
 *
 * items: an array of config items, at least one.
 */
CAPI_FUNC(MxSimulator*) MxSimulator_New(PyObject *args, PyObject *kw_args);

CAPI_FUNC(MxSimulator*) MxSimulator_Get();

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
CAPI_FUNC(HRESULT) MxSimulator_PollEvents();

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
CAPI_FUNC(HRESULT) MxSimulator_WaitEvents ();

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

CAPI_FUNC(HRESULT) MxSimulator_WaitEventsTimeout(double  timeout);


/**
 * This function posts an empty event from the current thread
 * to the event queue, causing waitEvents or waitEventsTimeout to return.
 */
CAPI_FUNC(HRESULT) MxSimulator_PostEmptyEvent();

/**
 * runs the event loop until window close
 */
CAPI_FUNC(HRESULT) MxSimulator_Run(double et);

/**
 * ipython version of the run loop. This checks the ipython context and lets
 * ipython process keyboard input, while we also run the simulator andx
 * process window messages.
 */
CAPI_FUNC(HRESULT) MxSimulator_InteractiveRun();

CAPI_FUNC(HRESULT) MxSimulator_Show();

CAPI_FUNC(HRESULT) MxSimulator_Close();

CAPI_FUNC(HRESULT) MxSimulator_Destroy();

CAPI_FUNC(HRESULT) MxSimulator_Redraw();



// internal method to initialize the simulator type.
HRESULT _MxSimulator_init(PyObject *o);


/**
 * This function sets the swap interval for the current OpenGL or OpenGL ES context, i.e. the number of screen updates to wait from the time glfwSwapBuffers was called before swapping the buffers and returning. This is sometimes called vertical synchronization, vertical retrace synchronization or just vsync.

A context that supports either of the WGL_EXT_swap_control_tear and GLX_EXT_swap_control_tear extensions also accepts negative swap intervals, which allows the driver to swap immediately even if a frame arrives a little bit late. You can check for these extensions with glfwExtensionSupported.

A context must be current on the calling thread. Calling this function without a current context will cause a GLFW_NO_CURRENT_CONTEXT error.

This function does not apply to Vulkan. If you are rendering with Vulkan, see the present mode of your swapchain instead.

Parameters
[in]    interval    The minimum number of screen updates to wait for until the buffers are swapped by glfwSwapBuffers.
Errors
Possible errors include GLFW_NOT_INITIALIZED, GLFW_NO_CURRENT_CONTEXT and GLFW_PLATFORM_ERROR.
Remarks
This function is not called during context creation, leaving the swap interval set to whatever is the default on that platform. This is done because some swap interval extensions used by GLFW do not allow the swap interval to be reset to zero once it has been set to a non-zero value.
Some GPU drivers do not honor the requested swap interval, either because of a user setting that overrides the application's request or due to bugs in the driver.
 */
HRESULT MxSimulator_SwapInterval(int si);


/**
 @brief OpenGL context configuration

 The created window is always with a double-buffered OpenGL context.

 @note This function is available only if Magnum is compiled with
 @ref MAGNUM_TARGET_GL enabled (done by default). See @ref building-features
 for more information.

 @see @ref MxGlfwApplication(), @ref create(), @ref tryCreate()
 */
class MxSimulator::GLConfig {
public:
    /**
     * @brief Context flag
     *
     * @see @ref Flags, @ref setFlags(), @ref GL::Context::Flag
     */
    enum  Flag: uint32_t {
#ifndef MAGNUM_TARGET_GLES
        /**
         * Forward compatible context
         *
         * @requires_gl Core/compatibility profile distinction and forward
         *      compatibility applies only to desktop GL.
         */
        ForwardCompatible = 1 << 0,
#endif

#if defined(DOXYGEN_GENERATING_OUTPUT) || defined(GLFW_CONTEXT_NO_ERROR)
        /**
         * Specifies whether errors should be generated by the context.
         * If enabled, situations that would have generated errors instead
         * cause undefined behavior.
         *
         * @note Supported since GLFW 3.2.
         */
        NoError = 1 << 1,
#endif

        /**
         * Debug context. Enabled automatically if the
         * `--magnum-gpu-validation` @ref GL-Context-command-line "command-line option"
         * is present.
         */
        Debug = 1 << 2,

        Stereo = 1 << 3     /**< Stereo rendering */
    };

    /**
     * @brief Context flags
     *
     * @see @ref setFlags(), @ref GL::Context::Flags
     */
    typedef uint32_t Flags;

    explicit GLConfig();
    ~GLConfig();

    /** @brief Context flags */
    Flags flags() const { return _flags; }

    /**
     * @brief Set context flags
     * @return Reference to self (for method chaining)
     *
     * Default is @ref Flag::ForwardCompatible on desktop GL and no flags
     * on OpenGL ES.
     * @see @ref addFlags(), @ref clearFlags(), @ref GL::Context::flags()
     */
    GLConfig& setFlags(Flags flags) {
        _flags = flags;
        return *this;
    }

    /**
     * @brief Add context flags
     * @return Reference to self (for method chaining)
     *
     * Unlike @ref setFlags(), ORs the flags with existing instead of
     * replacing them. Useful for preserving the defaults.
     * @see @ref clearFlags()
     */
    GLConfig& addFlags(Flags flags) {
        _flags |= flags;
        return *this;
    }

    /**
     * @brief Clear context flags
     * @return Reference to self (for method chaining)
     *
     * Unlike @ref setFlags(), ANDs the inverse of @p flags with existing
     * instead of replacing them. Useful for removing default flags.
     * @see @ref addFlags()
     */
    GLConfig& clearFlags(Flags flags) {
        _flags &= ~flags;
        return *this;
    }

    /** @brief Context version */
    GL::Version version() const { return _version; }

    /**
     * @brief Set context version
     *
     * If requesting version greater or equal to OpenGL 3.2, core profile
     * is used. The created context will then have any version which is
     * backwards-compatible with requested one. Default is
     * @ref GL::Version::None, i.e. any provided version is used.
     */
    GLConfig& setVersion(GL::Version version) {
        _version = version;
        return *this;
    }

    /** @brief Color buffer size */
    Vector4i colorBufferSize() const { return _colorBufferSize; }

    /**
     * @brief Set color buffer size
     *
     * Default is @cpp {8, 8, 8, 0} @ce (8-bit-per-channel RGB, no alpha).
     * @see @ref setDepthBufferSize(), @ref setStencilBufferSize()
     */
    GLConfig& setColorBufferSize(const Vector4i& size) {
        _colorBufferSize = size;
        return *this;
    }

    /** @brief Depth buffer size */
    Int depthBufferSize() const { return _depthBufferSize; }

    /**
     * @brief Set depth buffer size
     *
     * Default is @cpp 24 @ce bits.
     * @see @ref setColorBufferSize(), @ref setStencilBufferSize()
     */
    GLConfig& setDepthBufferSize(Int size) {
        _depthBufferSize = size;
        return *this;
    }

    /** @brief Stencil buffer size */
    Int stencilBufferSize() const { return _stencilBufferSize; }

    /**
     * @brief Set stencil buffer size
     *
     * Default is @cpp 0 @ce bits (i.e., no stencil buffer).
     * @see @ref setColorBufferSize(), @ref setDepthBufferSize()
     */
    GLConfig& setStencilBufferSize(Int size) {
        _stencilBufferSize = size;
        return *this;
    }

    /** @brief Sample count */
    Int sampleCount() const { return _sampleCount; }

    /**
     * @brief Set sample count
     * @return Reference to self (for method chaining)
     *
     * Default is @cpp 0 @ce, thus no multisampling. The actual sample
     * count is ignored, GLFW either enables it or disables. See also
     * @ref GL::Renderer::Feature::Multisampling.
     */
    GLConfig& setSampleCount(Int count) {
        _sampleCount = count;
        return *this;
    }

    /** @brief sRGB-capable default framebuffer */
    bool isSrgbCapable() const { return _srgbCapable; }

    /**
     * @brief Set sRGB-capable default framebuffer
     *
     * Default is @cpp false @ce. See also
     * @ref GL::Renderer::Feature::FramebufferSrgb.
     * @return Reference to self (for method chaining)
     */
    GLConfig& setSrgbCapable(bool enabled) {
        _srgbCapable = enabled;
        return *this;
    }


private:
    Vector4i _colorBufferSize;
    Int _depthBufferSize, _stencilBufferSize;
    Int _sampleCount;
    GL::Version _version;
    Flags _flags;
    bool _srgbCapable;
};


PyObject *MxSystem_CameraRotate(PyObject *self, PyObject *args, PyObject *kwargs);
PyObject *MxSystem_ContextRelease(PyObject *self);
PyObject *MxSystem_ContextMakeCurrent(PyObject *self);
PyObject *MxSystem_ContextHasCurrent(PyObject *self);

/**
 * main simulator init method
 */
PyObject *MxSimulator_Init(PyObject *self, PyObject *args, PyObject *kwargs);

// const Vector3 &origin, const Vector3 &dim,
// int nParticles, double dt = 0.005, float temp = 100

CAPI_FUNC(int) universe_init(const MxUniverseConfig &conf);


#endif /* SRC_MXSIMULATOR_H_ */
