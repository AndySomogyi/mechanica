/*
 * MxGlfwApplication.h
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 *
 *

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
              Vladimír Vondruš <mosra@centrum.cz>
    Copyright © 2016 Jonathan Hale <squareys@googlemail.com>
    Copyright © 2019, 2020 Marco Melorio <m.melorio@icloud.com>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

 */

#ifndef SRC_MXGLFWAPPLICATION_H_
#define SRC_MXGLFWAPPLICATION_H_

#include <mechanica_private.h>
#include <rendering/MxApplication.h>
#include <MxSimulator.h>
#include <Magnum/Platform/GlfwApplication.h>


using namespace Magnum::Platform;
using namespace Magnum;



/**
 * Copy of Magnum GLFW application, but this version does not create a window.
 */
//class MxGlfwApplication :
//        public MxApplication,
//        Magnum::Platform::MxGlfwApplication {




/** @file
 * @brief Class @ref Magnum::Platform::MxGlfwApplication, macro @ref MAGNUM_GLFWAPPLICATION_MAIN()
 */



/** @nosubgrouping
@brief GLFW application

@m_keywords{Application}

Application using the [GLFW](http://glfw.org) toolkit. Supports keyboard and
mouse handling with support for changing cursor and mouse tracking and warping.
Available on all platforms where GLFW is ported.

@m_class{m-block m-success}

@thirdparty This library makes use of [GLFW](https://www.glfw.org/), released
    under the @m_class{m-label m-success} **zlib/libpng license**
    ([license text](https://www.glfw.org/license.html),
    [choosealicense.com](https://choosealicense.com/licenses/zlib/)).
    Attribution is appreciated but not required.

@section Platform-MxGlfwApplication-bootstrap Bootstrap application

Fully contained base application using @ref MxGlfwApplication along with CMake
setup is available in `base-glfw` branch of
[Magnum Bootstrap](https://github.com/mosra/magnum-bootstrap) repository,
download it as [tar.gz](https://github.com/mosra/magnum-bootstrap/archive/base-glfw.tar.gz)
or [zip](https://github.com/mosra/magnum-bootstrap/archive/base-glfw.zip) file.
After extracting the downloaded archive you can build and run the application
with these four commands:

@code{.sh}
mkdir build && cd build
cmake ..
cmake --build .
./src/MyApplication # or ./src/Debug/MyApplication
@endcode

See @ref cmake for more information.

@section Platform-MxGlfwApplication-usage General usage

This application library depends on the [GLFW](http://glfw.org) library and is
built if `WITH_GLFWAPPLICATION` is enabled when building Magnum. To use this
library with CMake, put
[FindGLFW.cmake](https://github.com/mosra/magnum/blob/master/modules/FindGLFW.cmake)
into your `modules/` directory, request the `MxGlfwApplication` component of the
`Magnum` package and link to the `Magnum::MxGlfwApplication` target:

@code{.cmake}
find_package(Magnum REQUIRED MxGlfwApplication)

# ...
target_link_libraries(your-app PRIVATE Magnum::MxGlfwApplication)
@endcode

Additionally, if you're using Magnum as a CMake subproject, bundle the
[glfw repository](https://github.com/glfw/glfw) and do the following
 * *before* calling @cmake find_package() @ce to ensure it's enabled, as the
library is not built by default. If you want to use system-installed GLFW, omit
the first part and point `CMAKE_PREFIX_PATH` to its installation dir if
necessary.

@code{.cmake}
add_subdirectory(glfw)

set(WITH_GLFWAPPLICATION ON CACHE BOOL "" FORCE)
add_subdirectory(magnum EXCLUDE_FROM_ALL)
@endcode

If no other application is requested, you can also use the generic
`Magnum::Application` alias to simplify porting. Again, see @ref building and
@ref cmake for more information.

In C++ code you need to implement at least @ref drawEvent() to be able to draw
on the screen. The subclass can be then used directly in @cpp main() @ce
--- see convenience macro @ref MAGNUM_GLFWAPPLICATION_MAIN(). See @ref platform
for more information.

@code{.cpp}
class MyApplication: public Platform::MxGlfwApplication {
    // implement required methods...
};
MAGNUM_GLFWAPPLICATION_MAIN(MyApplication)
@endcode

If no other application header is included, this class is also aliased to
@cpp Platform::Application @ce and the macro is aliased to
@cpp MAGNUM_APPLICATION_MAIN() @ce to simplify porting.

@section Platform-MxGlfwApplication-dpi DPI awareness

DPI awareness behavior is consistent with @ref Sdl2Application except that iOS
or Emscripten specifics don't apply here. In addition, on Windows, GLFW is
implicitly advertising DPI awareness, so the manifest file described in
@ref platforms-windows-hidpi doesn't necessarily need to be supplied. See
@ref Platform-Sdl2Application-dpi "Sdl2Application DPI awareness documentation"
for more information.
 */
class MxGlfwApplication : public MxApplication {
public:
    /** @brief Application arguments */
    struct Arguments {
        /** @brief Constructor */
        /*implicit*/ constexpr Arguments(int& argc, char** argv) noexcept: argc{argc}, argv{argv} {}

        int& argc;      /**< @brief Argument count */
        char** argv;    /**< @brief Argument values */
    };

    typedef MxSimulator::Config Configuration;
    typedef MxSimulator::GLConfig GLConfig;

    class ExitEvent;
    class ViewportEvent;
    class InputEvent;
    class KeyEvent;
    class MouseEvent;
    class MouseMoveEvent;
    class MouseScrollEvent;
    class TextInputEvent;

#ifdef MAGNUM_TARGET_GL
    /**
     * @brief Construct with given configuration for OpenGL context
     * @param arguments         Application arguments
     * @param configuration     Application configuration
     * @param GLConfig   OpenGL context configuration
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
    explicit MxGlfwApplication(const Arguments& arguments, const Configuration& configuration, const GLConfig& GLConfig);
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
     * @ref MxGlfwApplication(const Arguments&, const Configuration&, const GLConfig&)
     * with default-constructed @ref GLConfig.
     *
     * See also @ref building-features for more information.
     */
    explicit MxGlfwApplication(const Arguments& arguments, const Configuration& configuration);

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
    explicit MxGlfwApplication(const Arguments& arguments, NoCreateT);

    /** @brief Copying is not allowed */
    MxGlfwApplication(const MxGlfwApplication&) = delete;

    /** @brief Moving is not allowed */
    MxGlfwApplication(MxGlfwApplication&&) = delete;

    /** @brief Copying is not allowed */
    MxGlfwApplication& operator=(const MxGlfwApplication&) = delete;

    /** @brief Moving is not allowed */
    MxGlfwApplication& operator=(MxGlfwApplication&&) = delete;

    /**
     * @brief Execute main loop
     * @return Value for returning from @cpp main() @ce
     *
     * See @ref MAGNUM_GLFWAPPLICATION_MAIN() for usage information.
     */
    int exec();

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
    bool mainLoopIteration();

    /**
     * @brief Exit application main loop
     * @param exitCode  The exit code the application should return
     */
    void exit(int exitCode = 0) {
        glfwSetWindowShouldClose(_window, true);
        _exitCode = exitCode;
    }

    /**
     * @brief Underlying window handle
     *
     * Use in case you need to call GLFW functionality directly.
     */
    GLFWwindow* window() { return _window; }


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
    virtual HRESULT pollEvents ();

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
    virtual HRESULT waitEvents ();

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

    virtual HRESULT waitEventsTimeout(double  timeout);


    /**
     * This function posts an empty event from the current thread
     * to the event queue, causing waitEvents or waitEventsTimeout to return.
     */
    virtual HRESULT postEmptyEvent();

protected:
    /* Nobody will need to have (and delete) MxGlfwApplication*, thus this is
           faster than public pure virtual destructor */
    virtual ~MxGlfwApplication();

#ifdef MAGNUM_TARGET_GL
    /**
     * @brief Create a window with given configuration for OpenGL context
     * @param configuration     Application configuration
     * @param GLConfig   OpenGL context configuration
     *
     * Must be called only if the context wasn't created by the constructor
     * itself, i.e. when passing @ref NoCreate to it. Error message is
     * printed and the program exits if the context cannot be created, see
     * @ref tryCreate() for an alternative.
     *
     * On desktop GL, if version is not specified in @p GLConfig,
     * the application first tries to create core context (OpenGL 3.2+) and
     * if that fails, falls back to compatibility OpenGL 2.1 context.
     *
     * @note This function is available only if Magnum is compiled with
     *      @ref MAGNUM_TARGET_GL enabled (done by default). See
     *      @ref building-features for more information.
     */
    void create(const Configuration& configuration, const GLConfig& GLConfig);
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
     * @ref create(const Configuration&, const GLConfig&) with
     * default-constructed @ref GLConfig.
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
     * Unlike @ref create(const Configuration&, const GLConfig&)
     * returns @cpp false @ce if the context cannot be created,
     * @cpp true @ce otherwise.
     *
     * @note This function is available only if Magnum is compiled with
     *      @ref MAGNUM_TARGET_GL enabled (done by default). See
     *      @ref building-features for more information.
     */
    bool tryCreate(const Configuration& configuration, const GLConfig& GLConfig);
#endif

    /**
     * @brief Try to create context with given configuration
     *
     * Unlike @ref create(const Configuration&) returns @cpp false @ce if
     * the context cannot be created, @cpp true @ce otherwise.
     */
    bool tryCreate(const Configuration& configuration);

    /** @{ @name Screen handling */

public:
    /**
     * @brief Window size
     *
     * Window size to which all input event coordinates can be related.
     * Note that, especially on HiDPI systems, it may be different from
     * @ref framebufferSize(). Expects that a window is already created.
     * See @ref Platform-MxGlfwApplication-dpi for more information.
     * @see @ref dpiScaling()
     */
    Vector2i windowSize() const;

    /**
     * @brief Set window size
     * @param size    The size, in screen coordinates
     * @m_since_latest
     *
     * To make the sizing work independently of the display DPI, @p size is
     * internally multiplied with @ref dpiScaling() before getting applied.
     * Expects that a window is already created.
     * @see @ref setMinWindowSize(), @ref setMaxWindowSize()
     */
    void setWindowSize(const Vector2i& size);

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

    /**
     * @brief Set window title
     * @m_since{2019,10}
     *
     * The @p title is expected to be encoded in UTF-8.
     */
    void setWindowTitle(const std::string& title);

#if GLFW_VERSION_MAJOR*100 + GLFW_VERSION_MINOR >= 302 || defined(DOXYGEN_GENERATING_OUTPUT)
    /**
     * @brief Set window icon
     * @m_since_latest
     *
     * The @p images are expected to be with origin at bottom left (which
     * is the default for imported images) and in one of
     * @ref PixelFormat::RGB8Unorm, @ref PixelFormat::RGB8Srgb,
     * @ref PixelFormat::RGBA8Unorm or @ref PixelFormat::RGBA8Srgb formats.
     * If you have just one image, you can use
     * @ref setWindowIcon(const ImageView2D&) instead.
     * @note Available since GLFW 3.2. The function has no effect on macOS
     *      / Wayland, see @m_class{m-doc-external} [glfwSetWindowIcon()](https://www.glfw.org/docs/latest/group__window.html#gadd7ccd39fe7a7d1f0904666ae5932dc5)
     *      for more information.
     */
    void setWindowIcon(std::initializer_list<ImageView2D> images);

    /**
     * @overload
     * @m_since_latest
     */
    void setWindowIcon(const ImageView2D& image);
#endif

    /**
     * @brief Swap buffers
     *
     * Paints currently rendered framebuffer on screen.
     */
    void swapBuffers() { glfwSwapBuffers(_window); }

    /**
     * @brief Set swap interval
     *
     * Set @cpp 0 @ce for no VSync, @cpp 1 @ce for enabled VSync. Some
     * platforms support @cpp -1 @ce for late swap tearing. Default is
     * driver-dependent.
     *
     * @note Unlike SDL2, GLFW doesn't provide any getter for the swap
     *      interval, so this class doesn't provide any equivalent to
     *      @ref Sdl2Application::swapInterval().
     */
    void setSwapInterval(Int interval);

    /** @copydoc Sdl2Application::redraw() */
    void redraw();

private:
    /**
     * @brief Viewport event
     *
     * Called when window size changes. The default implementation does
     * nothing. If you want to respond to size changes, you should pass the
     * new size to @ref GL::DefaultFramebuffer::setViewport() (if using
     * OpenGL) and possibly elsewhere (to
     * @ref SceneGraph::Camera::setViewport(), other framebuffers...).
     *
     * Note that this function might not get called at all if the window
     * size doesn't change. You should configure the initial state of your
     * cameras, framebuffers etc. in application constructor rather than
     * relying on this function to be called. Size of the window can be
     * retrieved using @ref windowSize(), size of the backing framebuffer
     * via @ref framebufferSize() and DPI scaling using @ref dpiScaling().
     * See @ref Platform-MxGlfwApplication-dpi for detailed info about these
     * values.
     */
    virtual void viewportEvent(ViewportEvent& event);

#ifdef MAGNUM_BUILD_DEPRECATED
    /** @brief @copybrief viewportEvent(ViewportEvent&)
     * @m_deprecated_since{2018,10} Use @ref viewportEvent(ViewportEvent&)
     *      instead. To preserve backwards compatibility, this function is
     *      called from @ref viewportEvent(ViewportEvent&) with
     *      @ref ViewportEvent::windowSize() passed to @p size. Overriding
     *      the new function will cause this function to not be called
     *      anymore.
     */
    virtual CORRADE_DEPRECATED("use viewportEvent(ViewportEvent&) instead") void viewportEvent(const Vector2i& size);
#endif

    /** @copydoc Sdl2Application::drawEvent() */
    virtual void drawEvent() {};

    /* Since 1.8.17, the original short-hand group closing doesn't work
           anymore. FFS. */
    /**
     * @}
     */

    /** @{ @name Keyboard handling */

    /** @copydoc Sdl2Application::keyPressEvent() */
    virtual void keyPressEvent(KeyEvent& event);

    /** @copydoc Sdl2Application::keyReleaseEvent() */
    virtual void keyReleaseEvent(KeyEvent& event);

    /* Since 1.8.17, the original short-hand group closing doesn't work
           anymore. FFS. */
    /**
     * @}
     */

    /** @{ @name Mouse handling */

public:
    /**
     * @brief Cursor type
     * @m_since_latest
     *
     * @see @ref setCursor()
     */
    enum class Cursor: UnsignedInt {
        Arrow,          /**< Arrow */
        TextInput,      /**< Text input */
        Crosshair,      /**< Crosshair */

        /* Checking for GLFW_RESIZE_NWSE_CURSOR being defined instead of a
               version check because older Git clones have version set to 3.4
               but don't contain those defines. All new cursors were added in
               the same commit, so it's okay to test for just one define. */
#if defined(DOXYGEN_GENERATING_OUTPUT) || defined(GLFW_RESIZE_NWSE_CURSOR)
        /**
         * Double arrow pointing northwest and southeast
         * @note Available since GLFW 3.4.
         */
        ResizeNWSE,

        /**
         * Double arrow pointing northeast and southwest
         * @note Available since GLFW 3.4.
         */
        ResizeNESW,
#endif

        ResizeWE,       /**< Double arrow pointing west and east */
        ResizeNS,       /**< Double arrow pointing north and south */

#if defined(DOXYGEN_GENERATING_OUTPUT) || defined(GLFW_RESIZE_NWSE_CURSOR)
        /**
         * Four pointed arrow pointing north, south, east, and west
         * @note Available since GLFW 3.4.
         */
        ResizeAll,

        /**
         * Slashed circle or crossbones
         * @note Available since GLFW 3.4.
         */
        No,
#endif

        Hand,           /**< Hand */
        Hidden,         /**< Hidden */
        HiddenLocked    /**< Hidden and locked */
    };

    /**
     * @brief Set cursor type
     * @m_since_latest
     *
     * Default is @ref Cursor::Arrow.
     */
    void setCursor(Cursor cursor);

    /**
     * @brief Get current cursor type
     * @m_since_latest
     */
    Cursor cursor();

    /**
     * @brief Warp mouse cursor to given coordinates
     * @m_since_latest
     */
    void warpCursor(const Vector2i& position) {
        glfwSetCursorPos(_window, Double(position.x()), Double(position.y()));
    }

private:
    /** @copydoc Sdl2Application::mousePressEvent() */
    virtual void mousePressEvent(MouseEvent& event);

    /** @copydoc Sdl2Application::mouseReleaseEvent() */
    virtual void mouseReleaseEvent(MouseEvent& event);

    /**
     * @brief Mouse move event
     *
     * Called when any mouse button is pressed and mouse is moved. Default
     * implementation does nothing.
     */
    virtual void mouseMoveEvent(MouseMoveEvent& event);

    /** @copydoc Sdl2Application::mouseScrollEvent() */
    virtual void mouseScrollEvent(MouseScrollEvent& event);

    /* Since 1.8.17, the original short-hand group closing doesn't work
           anymore. FFS. */
    /**
     * @}
     */

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
    /**
     * @brief Text input event
     *
     * Called when text input is active and the text is being input.
     * @see @ref isTextInputActive()
     */
    virtual void textInputEvent(TextInputEvent& event);

    /* Since 1.8.17, the original short-hand group closing doesn't work
           anymore. FFS. */
    /**
     * @}
     */

    /** @{ @name Special events */

    /**
     * @brief Exit event
     *
     * If implemented, it allows the application to react to an application
     * exit (for example to save its internal state) and suppress it as
     * well (for example to show a exit confirmation dialog). The default
     * implementation calls @ref ExitEvent::setAccepted() on @p event,
     * which tells the application that it's safe to exit.
     */
    virtual void exitEvent(ExitEvent& event);

    /* Since 1.8.17, the original short-hand group closing doesn't work
           anymore. FFS. */
    /**
     * @}
     */

private:
    enum class Flag: UnsignedByte;
    typedef Containers::EnumSet<Flag> Flags;
    CORRADE_ENUMSET_FRIEND_OPERATORS(Flags)

    void setupCallbacks();

    GLFWcursor* _cursors[8]{};
    Cursor _cursor = Cursor::Arrow;

    /* These are saved from command-line arguments */
    bool _verboseLog{};
    MxSimulator::DpiScalingPolicy _commandLineDpiScalingPolicy{};
    Vector2 _commandLineDpiScaling;

    Vector2 _dpiScaling;
    GLFWwindow* _window{nullptr};
    Flags _flags;
#ifdef MAGNUM_TARGET_GL
    Containers::Pointer<Platform::GLContext> _context;
#endif
    int _exitCode = 0;

    Vector2i _minWindowSize, _maxWindowSize;
    Vector2i _previousMouseMovePosition{-1};
};







/**
@brief Exit event

@see @ref exitEvent()
 */
class MxGlfwApplication::ExitEvent {
public:
    /** @brief Copying is not allowed */
    ExitEvent(const ExitEvent&) = delete;

    /** @brief Moving is not allowed */
    ExitEvent(ExitEvent&&) = delete;

    /** @brief Copying is not allowed */
    ExitEvent& operator=(const ExitEvent&) = delete;

    /** @brief Moving is not allowed */
    ExitEvent& operator=(ExitEvent&&) = delete;

    /** @brief Whether the event is accepted */
    bool isAccepted() const { return _accepted; }

    /**
     * @brief Set event as accepted
     *
     * If the event is ignored (i.e., not set as accepted) in
     * @ref exitEvent(), the application won't exit. Default implementation
     * of @ref exitEvent() accepts the event.
     */
    void setAccepted(bool accepted = true) { _accepted = accepted; }

private:
    friend MxGlfwApplication;

    explicit ExitEvent(): _accepted(false) {}

    bool _accepted;
};

/**
@brief Viewport event

@see @ref viewportEvent()
 */
class MxGlfwApplication::ViewportEvent {
public:
    /** @brief Copying is not allowed */
    ViewportEvent(const ViewportEvent&) = delete;

    /** @brief Moving is not allowed */
    ViewportEvent(ViewportEvent&&) = delete;

    /** @brief Copying is not allowed */
    ViewportEvent& operator=(const ViewportEvent&) = delete;

    /** @brief Moving is not allowed */
    ViewportEvent& operator=(ViewportEvent&&) = delete;

    /**
     * @brief Window size
     *
     * On some platforms with HiDPI displays, window size can be different
     * from @ref framebufferSize(). See @ref Platform-MxGlfwApplication-dpi
     * for more information.
     * @see @ref MxGlfwApplication::windowSize()
     */
    Vector2i windowSize() const { return _windowSize; }

#if defined(MAGNUM_TARGET_GL) || defined(DOXYGEN_GENERATING_OUTPUT)
    /**
     * @brief Framebuffer size
     *
     * On some platforms with HiDPI displays, framebuffer size can be
     * different from @ref windowSize(). See
     * @ref Platform-MxGlfwApplication-dpi for more information.
     *
     * @note This function is available only if Magnum is compiled with
     *      @ref MAGNUM_TARGET_GL enabled (done by default). See
     *      @ref building-features for more information.
     *
     * @see @ref MxGlfwApplication::framebufferSize(), @ref dpiScaling()
     */
    Vector2i framebufferSize() const { return _framebufferSize; }
#endif

    /**
     * @brief DPI scaling
     *
     * On some platforms moving an app between displays can result in DPI
     * scaling value being changed in tandem with a window/framebuffer
     * size. Simply resizing a window doesn't change the DPI scaling value.
     * See @ref Platform-MxGlfwApplication-dpi for more information.
     * @see @ref MxGlfwApplication::dpiScaling(), @ref framebufferSize()
     */
    Vector2 dpiScaling() const { return _dpiScaling; }

private:
    friend MxGlfwApplication;

    explicit ViewportEvent(const Vector2i& windowSize,
#ifdef MAGNUM_TARGET_GL
            const Vector2i& framebufferSize,
#endif
            const Vector2& dpiScaling): _windowSize{windowSize},
#ifdef MAGNUM_TARGET_GL
                    _framebufferSize{framebufferSize},
#endif
                    _dpiScaling{dpiScaling} {}

                    const Vector2i _windowSize;
#ifdef MAGNUM_TARGET_GL
                    const Vector2i _framebufferSize;
#endif
                    const Vector2 _dpiScaling;
};

/**
@brief Base for input events

@see @ref KeyEvent, @ref MouseEvent, @ref MouseMoveEvent, @ref keyPressEvent(),
    @ref mousePressEvent(), @ref mouseReleaseEvent(), @ref mouseMoveEvent()
 */
class MxGlfwApplication::InputEvent {
public:
    /**
     * @brief Modifier
     *
     * @see @ref Modifiers, @ref KeyEvent::modifiers(),
     *      @ref MouseEvent::modifiers()
     */
    enum class Modifier: Int {
        /**
         * Shift
         *
         * @see @ref KeyEvent::Key::LeftShift, @ref KeyEvent::Key::RightShift
         */
        Shift = GLFW_MOD_SHIFT,

                /**
                 * Ctrl
                 *
                 * @see @ref KeyEvent::Key::LeftCtrl, @ref KeyEvent::Key::RightCtrl
                 */
                Ctrl = GLFW_MOD_CONTROL,

                /**
                 * Alt
                 *
                 * @see @ref KeyEvent::Key::LeftAlt, @ref KeyEvent::Key::RightAlt
                 */
                Alt = GLFW_MOD_ALT,

                /**
                 * Super key (Windows/⌘)
                 *
                 * @see @ref KeyEvent::Key::LeftSuper, @ref KeyEvent::Key::RightSuper
                 */
                Super = GLFW_MOD_SUPER
    };

    /**
     * @brief Set of modifiers
     *
     * @see @ref KeyEvent::modifiers(), @ref MouseEvent::modifiers(),
     *      @ref MouseMoveEvent::modifiers()
     */
    typedef Containers::EnumSet<Modifier> Modifiers;

    /** @brief Copying is not allowed */
    InputEvent(const InputEvent&) = delete;

    /** @brief Moving is not allowed */
    InputEvent(InputEvent&&) = delete;

    /** @brief Copying is not allowed */
    InputEvent& operator=(const InputEvent&) = delete;

    /** @brief Moving is not allowed */
    InputEvent& operator=(InputEvent&&) = delete;

    /** @copydoc Sdl2Application::InputEvent::setAccepted() */
    void setAccepted(bool accepted = true) { _accepted = accepted; }

    /** @copydoc Sdl2Application::InputEvent::isAccepted() */
    bool isAccepted() const { return _accepted; }

protected:
    explicit InputEvent(): _accepted(false) {}

    ~InputEvent() = default;

private:
    bool _accepted;
};

CORRADE_ENUMSET_OPERATORS(MxGlfwApplication::InputEvent::Modifiers)

/**
@brief Key event

@see @ref keyPressEvent(), @ref keyReleaseEvent()
 */
class MxGlfwApplication::KeyEvent: public MxGlfwApplication::InputEvent {
public:
    /**
     * @brief Key
     *
     * @see @ref key()
     */
    enum class Key: Int {
        Unknown = GLFW_KEY_UNKNOWN,         /**< Unknown key */

                /**
                 * Left Shift
                 *
                 * @see @ref InputEvent::Modifier::Shift
                 */
                LeftShift = GLFW_KEY_LEFT_SHIFT,

                /**
                 * Right Shift
                 *
                 * @see @ref InputEvent::Modifier::Shift
                 */
                RightShift = GLFW_KEY_RIGHT_SHIFT,

                /**
                 * Left Ctrl
                 *
                 * @see @ref InputEvent::Modifier::Ctrl
                 */
                LeftCtrl = GLFW_KEY_LEFT_CONTROL,

                /**
                 * Right Ctrl
                 *
                 * @see @ref InputEvent::Modifier::Ctrl
                 */
                RightCtrl = GLFW_KEY_RIGHT_CONTROL,

                /**
                 * Left Alt
                 *
                 * @see @ref InputEvent::Modifier::Alt
                 */
                LeftAlt = GLFW_KEY_LEFT_ALT,

                /**
                 * Right Alt
                 *
                 * @see @ref InputEvent::Modifier::Alt
                 */
                RightAlt = GLFW_KEY_RIGHT_ALT,

                /**
                 * Left Super key (Windows/⌘)
                 *
                 * @see @ref InputEvent::Modifier::Super
                 */
                LeftSuper = GLFW_KEY_LEFT_SUPER,

                /**
                 * Right Super key (Windows/⌘)
                 *
                 * @see @ref InputEvent::Modifier::Super
                 */
                RightSuper = GLFW_KEY_RIGHT_SUPER,

                /* no equivalent for Sdl2Application's AltGr */

                Enter = GLFW_KEY_ENTER,             /**< Enter */
                Esc = GLFW_KEY_ESCAPE,              /**< Escape */

                Up = GLFW_KEY_UP,                   /**< Up arrow */
                Down = GLFW_KEY_DOWN,               /**< Down arrow */
                Left = GLFW_KEY_LEFT,               /**< Left arrow */
                Right = GLFW_KEY_RIGHT,             /**< Right arrow */
                Home = GLFW_KEY_HOME,               /**< Home */
                End = GLFW_KEY_END,                 /**< End */
                PageUp = GLFW_KEY_PAGE_UP,          /**< Page up */
                PageDown = GLFW_KEY_PAGE_DOWN,      /**< Page down */
                Backspace = GLFW_KEY_BACKSPACE,     /**< Backspace */
                Insert = GLFW_KEY_INSERT,           /**< Insert */
                Delete = GLFW_KEY_DELETE,           /**< Delete */

                F1 = GLFW_KEY_F1,                   /**< F1 */
                F2 = GLFW_KEY_F2,                   /**< F2 */
                F3 = GLFW_KEY_F3,                   /**< F3 */
                F4 = GLFW_KEY_F4,                   /**< F4 */
                F5 = GLFW_KEY_F5,                   /**< F5 */
                F6 = GLFW_KEY_F6,                   /**< F6 */
                F7 = GLFW_KEY_F7,                   /**< F7 */
                F8 = GLFW_KEY_F8,                   /**< F8 */
                F9 = GLFW_KEY_F9,                   /**< F9 */
                F10 = GLFW_KEY_F10,                 /**< F10 */
                F11 = GLFW_KEY_F11,                 /**< F11 */
                F12 = GLFW_KEY_F12,                 /**< F12 */

                Space = GLFW_KEY_SPACE,             /**< Space */
                Tab = GLFW_KEY_TAB,                 /**< Tab */

                /**
                 * Quote (<tt>'</tt>)
                 * @m_since_latest
                 */
                Quote = GLFW_KEY_APOSTROPHE,

                Comma = GLFW_KEY_COMMA,             /**< Comma */
                Period = GLFW_KEY_PERIOD,           /**< Period */
                Minus = GLFW_KEY_MINUS,             /**< Minus */
                /* Note: This may only be represented as SHIFT + = */
                Plus = '+',                         /**< Plus */
                Slash = GLFW_KEY_SLASH,             /**< Slash */
                /* Note: This may only be represented as SHIFT + 5 */
                Percent = '%',                      /**< Percent */
                Semicolon = GLFW_KEY_SEMICOLON,     /**< Semicolon (`;`) */

#ifdef MAGNUM_BUILD_DEPRECATED
                /** Semicolon (`;`)
                 * @m_deprecated_since{2019,01} Use @ref Key::Semicolon instead.
                 */
                Smicolon CORRADE_DEPRECATED_ENUM("use Key::Semicolon instead") = Semicolon,
#endif

                Equal = GLFW_KEY_EQUAL,             /**< Equal */

                /**
                 * Left bracket (`[`)
                 * @m_since_latest
                 */
                LeftBracket = GLFW_KEY_LEFT_BRACKET,

                /**
                 * Right bracket (`]`)
                 * @m_since_latest
                 */
                RightBracket = GLFW_KEY_RIGHT_BRACKET,

                /**
                 * Backslash (`\`)
                 * @m_since_latest
                 */
                Backslash = GLFW_KEY_BACKSLASH,

                /**
                 * Backquote (<tt>`</tt>)
                 * @m_since_latest
                 */
                Backquote = GLFW_KEY_GRAVE_ACCENT,

                /**
                 * Non-US \#1
                 * @m_since_latest
                 */
                World1 = GLFW_KEY_WORLD_1,

                /**
                 * Non-US \#2
                 * @m_since_latest
                 */
                World2 = GLFW_KEY_WORLD_2,

                Zero = GLFW_KEY_0,                  /**< Zero */
                One = GLFW_KEY_1,                   /**< One */
                Two = GLFW_KEY_2,                   /**< Two */
                Three = GLFW_KEY_3,                 /**< Three */
                Four = GLFW_KEY_4,                  /**< Four */
                Five = GLFW_KEY_5,                  /**< Five */
                Six = GLFW_KEY_6,                   /**< Six */
                Seven = GLFW_KEY_7,                 /**< Seven */
                Eight = GLFW_KEY_8,                 /**< Eight */
                Nine = GLFW_KEY_9,                  /**< Nine */

                A = GLFW_KEY_A,                     /**< Letter A */
                B = GLFW_KEY_B,                     /**< Letter B */
                C = GLFW_KEY_C,                     /**< Letter C */
                D = GLFW_KEY_D,                     /**< Letter D */
                E = GLFW_KEY_E,                     /**< Letter E */
                F = GLFW_KEY_F,                     /**< Letter F */
                G = GLFW_KEY_G,                     /**< Letter G */
                H = GLFW_KEY_H,                     /**< Letter H */
                I = GLFW_KEY_I,                     /**< Letter I */
                J = GLFW_KEY_J,                     /**< Letter J */
                K = GLFW_KEY_K,                     /**< Letter K */
                L = GLFW_KEY_L,                     /**< Letter L */
                M = GLFW_KEY_M,                     /**< Letter M */
                N = GLFW_KEY_N,                     /**< Letter N */
                O = GLFW_KEY_O,                     /**< Letter O */
                P = GLFW_KEY_P,                     /**< Letter P */
                Q = GLFW_KEY_Q,                     /**< Letter Q */
                R = GLFW_KEY_R,                     /**< Letter R */
                S = GLFW_KEY_S,                     /**< Letter S */
                T = GLFW_KEY_T,                     /**< Letter T */
                U = GLFW_KEY_U,                     /**< Letter U */
                V = GLFW_KEY_V,                     /**< Letter V */
                W = GLFW_KEY_W,                     /**< Letter W */
                X = GLFW_KEY_X,                     /**< Letter X */
                Y = GLFW_KEY_Y,                     /**< Letter Y */
                Z = GLFW_KEY_Z,                     /**< Letter Z */

                CapsLock = GLFW_KEY_CAPS_LOCK,      /**< Caps lock */
                ScrollLock = GLFW_KEY_SCROLL_LOCK,  /**< Scroll lock */
                NumLock = GLFW_KEY_NUM_LOCK,        /**< Num lock */
                PrintScreen = GLFW_KEY_PRINT_SCREEN,/**< Print screen */
                Pause = GLFW_KEY_PAUSE,             /**< Pause */
                Menu = GLFW_KEY_MENU,               /**< Menu */

                NumZero = GLFW_KEY_KP_0,            /**< Numpad zero */
                NumOne = GLFW_KEY_KP_1,             /**< Numpad one */
                NumTwo = GLFW_KEY_KP_2,             /**< Numpad two */
                NumThree = GLFW_KEY_KP_3,           /**< Numpad three */
                NumFour = GLFW_KEY_KP_4,            /**< Numpad four */
                NumFive = GLFW_KEY_KP_5,            /**< Numpad five */
                NumSix = GLFW_KEY_KP_6,             /**< Numpad six */
                NumSeven = GLFW_KEY_KP_7,           /**< Numpad seven */
                NumEight = GLFW_KEY_KP_8,           /**< Numpad eight */
                NumNine = GLFW_KEY_KP_9,            /**< Numpad nine */
                NumDecimal = GLFW_KEY_KP_DECIMAL,   /**< Numpad decimal */
                NumDivide = GLFW_KEY_KP_DIVIDE,     /**< Numpad divide */
                NumMultiply = GLFW_KEY_KP_MULTIPLY, /**< Numpad multiply */
                NumSubtract = GLFW_KEY_KP_SUBTRACT, /**< Numpad subtract */
                NumAdd = GLFW_KEY_KP_ADD,           /**< Numpad add */
                NumEnter = GLFW_KEY_KP_ENTER,       /**< Numpad enter */
                NumEqual = GLFW_KEY_KP_EQUAL        /**< Numpad equal */
    };

#if defined(DOXYGEN_GENERATING_OUTPUT) || GLFW_VERSION_MAJOR*100 + GLFW_VERSION_MINOR >= 302
    /**
     * @brief Name for given key
     *
     * Human-readable localized UTF-8 name for given @p key, intended for
     * displaying to the user in e.g. key binding configuration. If there
     * is no name for given key, empty string is returned.
     * @see @ref keyName(Key)
     * @note Supported since GLFW 3.2.
     */
    static std::string keyName(Key key);
#endif

    /** @copydoc Sdl2Application::KeyEvent::key() */
    Key key() const { return _key; }

#if defined(DOXYGEN_GENERATING_OUTPUT) || GLFW_VERSION_MAJOR*100 + GLFW_VERSION_MINOR >= 302
    /**
     * @brief Key name
     *
     * Human-readable localized UTF-8 name for the key returned by
     * @ref key(), intended for displaying to the user in e.g.
     * key binding configuration. If there is no name for that key, empty
     * string is returned.
     * @see @ref keyName(Key)
     * @note Supported since GLFW 3.2.
     */
    std::string keyName() const;
#endif

    /** @brief Modifiers */
    Modifiers modifiers() const { return _modifiers; }

    /** @copydoc Sdl2Application::KeyEvent::isRepeated() */
    bool isRepeated() const { return _repeated; }

private:
    friend MxGlfwApplication;

    explicit KeyEvent(Key key, Modifiers modifiers, bool repeated): _key{key}, _modifiers{modifiers}, _repeated{repeated} {}

    const Key _key;
    const Modifiers _modifiers;
    const bool _repeated;
};

/**
@brief Mouse event

@see @ref MouseMoveEvent, @ref MouseScrollEvent, @ref mousePressEvent(),
    @ref mouseReleaseEvent()
 */
class MxGlfwApplication::MouseEvent: public MxGlfwApplication::InputEvent {
public:
    /**
     * @brief Mouse button
     *
     * @see @ref button()
     */
    enum class Button: int {
        Left = GLFW_MOUSE_BUTTON_LEFT,        /**< Left button */
                Middle = GLFW_MOUSE_BUTTON_MIDDLE,    /**< Middle button */
                Right = GLFW_MOUSE_BUTTON_RIGHT,      /**< Right button */
                Button1 = GLFW_MOUSE_BUTTON_1,        /**< Mouse button 1 */
                Button2 = GLFW_MOUSE_BUTTON_2,        /**< Mouse button 2 */
                Button3 = GLFW_MOUSE_BUTTON_3,        /**< Mouse button 3 */
                Button4 = GLFW_MOUSE_BUTTON_4,        /**< Mouse button 4 */
                Button5 = GLFW_MOUSE_BUTTON_5,        /**< Mouse button 5 */
                Button6 = GLFW_MOUSE_BUTTON_6,        /**< Mouse button 6 */
                Button7 = GLFW_MOUSE_BUTTON_7,        /**< Mouse button 7 */
                Button8 = GLFW_MOUSE_BUTTON_8,        /**< Mouse button 8 */
    };

    /** @brief Button */
    Button button() const { return _button; }

    /** @brief Position */
    Vector2i position() const { return _position; }

    /** @brief Modifiers */
    Modifiers modifiers() const { return _modifiers; }

private:
    friend MxGlfwApplication;

    explicit MouseEvent(Button button, const Vector2i& position, Modifiers modifiers): _button{button}, _position{position}, _modifiers{modifiers} {}

    const Button _button;
    const Vector2i _position;
    const Modifiers _modifiers;
};

/**
@brief Mouse move event

@see @ref MouseEvent, @ref MouseScrollEvent, @ref mouseMoveEvent()
 */
class MxGlfwApplication::MouseMoveEvent: public MxGlfwApplication::InputEvent {
public:
    /**
     * @brief Mouse button
     *
     * @see @ref Buttons, @ref buttons()
     */
    enum class Button: UnsignedInt {
        Left = 1 << GLFW_MOUSE_BUTTON_LEFT,     /**< Left button */
                Middle = 1 << GLFW_MOUSE_BUTTON_MIDDLE, /**< Middle button */
                Right = 1 << GLFW_MOUSE_BUTTON_RIGHT    /**< Right button */
    };

    /**
     * @brief Set of mouse buttons
     *
     * @see @ref buttons()
     */
    typedef Containers::EnumSet<Button> Buttons;

    /**
     * @brief Mouse buttons
     *
     * Lazily populated on first request.
     */
    Buttons buttons();

    /** @brief Position */
    Vector2i position() const { return _position; }

    /**
     * @brief Relative position
     * @m_since{2019,10}
     *
     * Position relative to previous move event. Unlike
     * @ref Sdl2Application, GLFW doesn't provide relative position
     * directly, so this is calculated explicitly as a delta from previous
     * move event position.
     */
    Vector2i relativePosition() const { return _relativePosition; }

    /**
     * @brief Modifiers
     *
     * Lazily populated on first request.
     */
    Modifiers modifiers();

private:
    friend MxGlfwApplication;

    explicit MouseMoveEvent(GLFWwindow* window, const Vector2i& position, const Vector2i& relativePosition): _window{window}, _position{position}, _relativePosition{relativePosition} {}

    GLFWwindow* const _window;
    const Vector2i _position, _relativePosition;
    Containers::Optional<Buttons> _buttons;
    Containers::Optional<Modifiers> _modifiers;
};

CORRADE_ENUMSET_OPERATORS(MxGlfwApplication::MouseMoveEvent::Buttons)

/**
@brief Mouse scroll event

@see @ref MouseEvent, @ref MouseMoveEvent, @ref mouseScrollEvent()
 */
class MxGlfwApplication::MouseScrollEvent: public MxGlfwApplication::InputEvent {
public:
    /** @brief Scroll offset */
    Vector2 offset() const { return _offset; }

    /**
     * @brief Position
     *
     * Lazily populated on first request.
     */
    Vector2i position();

    /**
     * @brief Modifiers
     *
     * Lazily populated on first request.
     */
    Modifiers modifiers();

private:
    friend MxGlfwApplication;

    explicit MouseScrollEvent(GLFWwindow* window, const Vector2& offset): _window{window}, _offset{offset} {}

    GLFWwindow* const _window;
    const Vector2 _offset;
    Containers::Optional<Vector2i> _position;
    Containers::Optional<Modifiers> _modifiers;
};

/**
@brief Text input event

@see @ref textInputEvent()
 */
class MxGlfwApplication::TextInputEvent {
public:
    /** @brief Copying is not allowed */
    TextInputEvent(const TextInputEvent&) = delete;

    /** @brief Moving is not allowed */
    TextInputEvent(TextInputEvent&&) = delete;

    /** @brief Copying is not allowed */
    TextInputEvent& operator=(const TextInputEvent&) = delete;

    /** @brief Moving is not allowed */
    TextInputEvent& operator=(TextInputEvent&&) = delete;

    /** @brief Whether the event is accepted */
    bool isAccepted() const { return _accepted; }

    /**
     * @brief Set event as accepted
     *
     * If the event is ignored (i.e., not set as accepted), it might be
     * propagated elsewhere, for example to another screen when using
     * @ref BasicScreenedApplication "ScreenedApplication". By default is
     * each event ignored and thus propagated.
     */
    void setAccepted(bool accepted = true) { _accepted = accepted; }

    /** @brief Input text in UTF-8 */
    Containers::ArrayView<const char> text() const { return _text; }

private:
    friend MxGlfwApplication;

    explicit TextInputEvent(Containers::ArrayView<const char> text): _text{text}, _accepted{false} {}

    const Containers::ArrayView<const char> _text;
    bool _accepted;
};




#endif /* SRC_MXGLFWAPPLICATION_H_ */
