/*
 * MxGlfwApplication.h
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#ifndef SRC_MXGLFWAPPLICATION_H_
#define SRC_MXGLFWAPPLICATION_H_


#include "MxApplication.h"
#include <Magnum/Platform/GlfwApplication.h>


using namespace Magnum::Platform;
using namespace Magnum;

/**
 * Copy of Magnum GLFW application, but this version does not create a window.
 */
class MxGlfwApplication {
    public:
        /** @brief Application arguments */
        struct Arguments {
            /** @brief Constructor */
            /*implicit*/ constexpr Arguments(int& argc, char** argv) noexcept: argc{argc}, argv{argv} {}

            int& argc;      /**< @brief Argument count */
            char** argv;    /**< @brief Argument values */
        };

        typedef Magnum::Platform::GlfwApplication::Configuration Configuration;
        typedef Magnum::Platform::GlfwApplication::GLConfiguration GLConfiguration;


        #ifdef MAGNUM_TARGET_GL
        /**
         * @brief Construct with given configuration for OpenGL context
         * @param arguments         Application arguments
         * @param configuration     Application configuration
         * @param glConfiguration   OpenGL context configuration
         *
         * Creates application with default or user-specified configuration.
         * See @ref Configuration for more information. The program exits if
         * the context cannot be created, see @ref tryCreate() for an
         * alternative.
         *
         * @note This function is available only if Magnum is compiled with
         *      @ref MAGNUM_TARGET_GL enabled (done by default). See
         *      @ref building-features for more information.
         */
        explicit MxGlfwApplication(const Arguments& arguments,
                const Configuration& configuration, const GLConfiguration& glConfiguration);
        #endif

        /**
         * @brief Construct with given configuration
         *
         * If @ref Configuration::WindowFlag::Contextless is present or Magnum
         * was not built with @ref MAGNUM_TARGET_GL, this creates a window
         * without any GPU context attached, leaving that part on the user.
         *
         * If none of the flags is present and Magnum was built with
         * @ref MAGNUM_TARGET_GL, this is equivalent to calling
         * @ref MxGlfwApplication(const Arguments&, const Configuration&, const GLConfiguration&)
         * with default-constructed @ref GLConfiguration.
         *
         * See also @ref building-features for more information.
         */
        explicit MxGlfwApplication(const Arguments& arguments,
                const Configuration& configuration);

        /**
         * @brief Construct with default configuration
         *
         * Equivalent to calling @ref MxGlfwApplication(const Arguments&, const Configuration&)
         * with default-constructed @ref Configuration.
         */
        explicit MxGlfwApplication(const Arguments& arguments);

        /**
         * @brief Construct without creating a window
         * @param arguments     Application arguments
         *
         * Unlike above, the window is not created and must be created later
         * with @ref create() or @ref tryCreate().
         */
        explicit MxGlfwApplication(const Arguments& arguments, Magnum::NoCreateT);

        /** @brief Copying is not allowed */
        MxGlfwApplication(const MxGlfwApplication&) = delete;

        /** @brief Moving is not allowed */
        MxGlfwApplication(MxGlfwApplication&&) = delete;

        /** @brief Copying is not allowed */
        MxGlfwApplication& operator=(const MxGlfwApplication&) = delete;

        /** @brief Moving is not allowed */
        MxGlfwApplication& operator=(MxGlfwApplication&&) = delete;


        /**
         * @brief Run one iteration of application main loop
         * @return @cpp false @ce if @ref exit() was called and the application
         *      should exit, @cpp true @ce otherwise
         * @m_since_latest
         *
         * Called internally from @ref exec(). If you want to have better
         * control over how the main loop behaves, you can call this function
         * yourself from your own `main()` function instead of it being called
         * automatically from @ref exec().
         */
        //bool mainLoopIteration();

        /**
         * @brief Exit application main loop
         * @param exitCode  The exit code the application should return
         *
         * TODO: need stuff to handle all windows
         */
        void exit(int exitCode = 0) {
            //glfwSetWindowShouldClose(_window, true);
            //_exitCode = exitCode;
        }



    protected:
        /* Nobody will need to have (and delete) MxGlfwApplication*, thus this is
           faster than public pure virtual destructor */
        ~MxGlfwApplication() {};

        #ifdef MAGNUM_TARGET_GL
        /**
         * @brief Create a window with given configuration for OpenGL context
         * @param configuration     Application configuration
         * @param glConfiguration   OpenGL context configuration
         *
         * Must be called only if the context wasn't created by the constructor
         * itself, i.e. when passing @ref NoCreate to it. Error message is
         * printed and the program exits if the context cannot be created, see
         * @ref tryCreate() for an alternative.
         *
         * On desktop GL, if version is not specified in @p glConfiguration,
         * the application first tries to create core context (OpenGL 3.2+) and
         * if that fails, falls back to compatibility OpenGL 2.1 context.
         *
         * @note This function is available only if Magnum is compiled with
         *      @ref MAGNUM_TARGET_GL enabled (done by default). See
         *      @ref building-features for more information.
         */
        void create(const Configuration& configuration, const GLConfiguration& glConfiguration);
        #endif

        /**
         * @brief Create a window with given configuration
         *
         * If @ref Configuration::WindowFlag::Contextless is present or Magnum
         * was not built with @ref MAGNUM_TARGET_GL, this creates a window
         * without any GPU context attached, leaving that part on the user.
         *
         * If none of the flags is present and Magnum was built with
         * @ref MAGNUM_TARGET_GL, this is equivalent to calling
         * @ref create(const Configuration&, const GLConfiguration&) with
         * default-constructed @ref GLConfiguration.
         *
         * See also @ref building-features for more information.
         */
        void create(const Configuration& configuration);

        /**
         * @brief Create a window with default configuration and OpenGL context
         *
         * Equivalent to calling @ref create(const Configuration&) with
         * default-constructed @ref Configuration.
         */
        void create();

        #ifdef MAGNUM_TARGET_GL
        /**
         * @brief Try to create context with given configuration for OpenGL context
         *
         * Unlike @ref create(const Configuration&, const GLConfiguration&)
         * returns @cpp false @ce if the context cannot be created,
         * @cpp true @ce otherwise.
         *
         * @note This function is available only if Magnum is compiled with
         *      @ref MAGNUM_TARGET_GL enabled (done by default). See
         *      @ref building-features for more information.
         */
        bool tryCreate(const Configuration& configuration, const GLConfiguration& glConfiguration);
        #endif

        /**
         * @brief Try to create context with given configuration
         *
         * Unlike @ref create(const Configuration&) returns @cpp false @ce if
         * the context cannot be created, @cpp true @ce otherwise.
         */
        bool tryCreate(const Configuration& configuration);

        /** @{ @name Screen handling */



        #if GLFW_VERSION_MAJOR*100 + GLFW_VERSION_MINOR >= 302 || defined(DOXYGEN_GENERATING_OUTPUT)
        /**
         * @brief Set window minimum size
         * @param size    The minimum size, in screen coordinates
         * @m_since{2019,10}
         *
         * If a value is set to @cpp -1 @ce, it will disable/remove the
         * corresponding limit. To make the sizing work independently of the
         * display DPI, @p size is internally multiplied with @ref dpiScaling()
         * before getting applied. Expects that a window is already created.
         * @note Supported since GLFW 3.2.
         * @see @ref setMaxWindowSize(), @ref setWindowSize()
         */
        void setMinWindowSize(const Vector2i& size = {-1, -1});

        /**
         * @brief Set window maximum size
         * @param size    The maximum size, in screen coordinates
         * @m_since{2019,10}
         *
         * If a value is set to @cpp -1 @ce, it will disable/remove the
         * corresponding limit. To make the sizing work independently of the
         * display DPI, @p size is internally multiplied with @ref dpiScaling()
         * before getting applied. Expects that a window is already created.
         * @note Supported since GLFW 3.2.
         * @see @ref setMinWindowSize(), @ref setMaxWindowSize()
         */
        void setMaxWindowSize(const Vector2i& size = {-1, -1});
        #endif

        #if defined(MAGNUM_TARGET_GL) || defined(DOXYGEN_GENERATING_OUTPUT)
        /**
         * @brief Framebuffer size
         *
         * Size of the default framebuffer. Note that, especially on HiDPI
         * systems, it may be different from @ref windowSize(). Expects that a
         * window is already created. See @ref Platform-MxGlfwApplication-dpi for
         * more information.
         *
         * @note This function is available only if Magnum is compiled with
         *      @ref MAGNUM_TARGET_GL enabled (done by default). See
         *      @ref building-features for more information.
         *
         * @see @ref dpiScaling()
         */
        Vector2i framebufferSize() const;
        #endif

        /**
         * @brief DPI scaling
         *
         * How the content should be scaled relative to system defaults for
         * given @ref windowSize(). If a window is not created yet, returns
         * zero vector, use @ref dpiScaling(const Configuration&) for
         * calculating a value independently. See @ref Platform-MxGlfwApplication-dpi
         * for more information.
         * @see @ref framebufferSize()
         */
        Vector2 dpiScaling() const { return _dpiScaling; }

        /**
         * @brief DPI scaling for given configuration
         *
         * Calculates DPI scaling that would be used when creating a window
         * with given @p configuration. Takes into account DPI scaling policy
         * and custom scaling specified on the command-line. See
         * @ref Platform-MxGlfwApplication-dpi for more information.
         */
        Vector2 dpiScaling(const Configuration& configuration);











        /** @{ @name Text input handling */
    public:
        /**
         * @brief Whether text input is active
         *
         * If text input is active, text input events go to
         * @ref textInputEvent().
         * @see @ref startTextInput(), @ref stopTextInput()
         */
        bool isTextInputActive() const;

        /**
         * @brief Start text input
         *
         * Starts text input that will go to @ref textInputEvent().
         * @see @ref stopTextInput(), @ref isTextInputActive()
         */
        void startTextInput();

        /**
         * @brief Stop text input
         *
         * Stops text input that went to @ref textInputEvent().
         * @see @ref startTextInput(), @ref isTextInputActive(),
         *      @ref textInputEvent()
         */
        void stopTextInput();




    private:
        enum class Flag: UnsignedByte;
        typedef Containers::EnumSet<Flag> Flags;
        CORRADE_ENUMSET_FRIEND_OPERATORS(Flags)





        /* These are saved from command-line arguments */
        bool _verboseLog{};
        Implementation::GlfwDpiScalingPolicy _commandLineDpiScalingPolicy{};
        Vector2 _commandLineDpiScaling;

        Vector2 _dpiScaling;
        //GLFWwindow* _window{nullptr};
        Flags _flags;
        #ifdef MAGNUM_TARGET_GL
        Containers::Pointer<Platform::GLContext> _context;
        #endif
        int _exitCode = 0;

        Vector2i _minWindowSize, _maxWindowSize;
        Vector2i _previousMouseMovePosition{-1};
};

#endif /* SRC_MXGLFWAPPLICATION_H_ */
