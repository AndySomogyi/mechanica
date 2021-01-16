
#include "MxGlfwApplication.h"
#include "MxKeyEvent.hpp"
#include <rendering/MxUniverseRenderer.h>

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


#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>

#if defined(_WIN32)
  #define GLFW_EXPOSE_NATIVE_WIN32
  #include <GLFW/glfw3native.h>
#endif

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


static Platform::GlfwApplication::Configuration confconf(const MxSimulator::Config &conf) {
    Platform::GlfwApplication::Configuration res;

    res.setSize(conf.windowSize(), conf.dpiScaling());
    res.setTitle(conf.title());
    res.setWindowFlags(Platform::GlfwApplication::Configuration::WindowFlag::Resizable);

    return res;
}


MxGlfwApplication::MxGlfwApplication(const Arguments &args) :
        Platform::GlfwApplication{args, NoCreate}
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

Magnum::GL::AbstractFramebuffer& MxGlfwApplication::framebuffer() {
    return GL::defaultFramebuffer;
}



void MxGlfwApplication::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);


    /* Draw particles */
    _ren->draw();

    swapBuffers();
    _timeline.nextFrame();
}

static MxGlfwApplication::Configuration magConf(const MxSimulator::Config &sc) {
    MxGlfwApplication::Configuration mc;

    mc.setTitle(sc.title());

    uint32_t wf = sc.windowFlags();

    if(wf & MxSimulator::AutoIconify) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::AutoIconify);
    }

    if(wf & MxSimulator::AlwaysOnTop) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::AlwaysOnTop);
    }

    if(wf & MxSimulator::AutoIconify) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::AutoIconify);
    }

    if(wf & MxSimulator::Borderless) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Borderless);
    }

    if(wf & MxSimulator::Contextless) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Contextless);
    }

    if(wf & MxSimulator::Focused) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Focused);
    }

    if(wf & MxSimulator::Fullscreen) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Fullscreen);
    }
    if(wf & MxSimulator::Hidden) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Hidden);
    }
    if(wf & MxSimulator::Maximized) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Maximized);
    }
    if(wf & MxSimulator::Minimized) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Minimized);
    }

    if(wf & MxSimulator::Resizable) {
        mc.addWindowFlags(MxGlfwApplication::Configuration::WindowFlag::Resizable);
    }


    return mc;
}

HRESULT MxGlfwApplication::createContext(const MxSimulator::Config &conf)
{
    const Vector2 dpiScaling = this->dpiScaling({});
    Configuration c = magConf(conf);
    c.setSize(conf.windowSize(), dpiScaling);

    GLConfiguration glConf;
    glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);

    glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE);
    
    bool b = tryCreate(c);

    if(!b) {
        return E_FAIL;
    }

    _win = new MxGlfwWindow(this->window());
    
    if(conf.windowFlags() & MxSimulator::WindowFlags::Focused) {
        glfwFocusWindow(this->window());
    }

    _ren = new MxUniverseRenderer{_win};

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



HRESULT MxGlfwApplication::messageLoop(double et)
{
    double initTime = _Engine.time * _Engine.dt;
    double endTime = std::numeric_limits<double>::infinity();
    
    if(et >= 0) {
        endTime = initTime + et;
    }
    
    std::cout << "MxGlfwApplication::messageLoop(" << et 
              << ") {now_time: " << initTime << ", end_time: " << endTime <<  "}" << std::endl;
    
    // process initial messages.
    GlfwApplication::mainLoopIteration();

    // show the window
    showWindow();
    
    // need to post an empty message for some reason.
    // if you start the app, the simulation loop won't start
    // until the moust moves or some event happens, so send a
    // empty event here to start it.
    glfwPostEmptyEvent();

#if defined(_WIN32)
    std::fprintf(stderr, "set forground window \n");
    GLFWwindow* wnd = window();
    HWND hwnd = glfwGetWin32Window(wnd);
    SetForegroundWindow(hwnd);
#endif
    
    HRESULT hr;

    // run while it's visible, process window messages
    while(_Engine.time * _Engine.dt < endTime &&
          GlfwApplication::mainLoopIteration() &&
          glfwGetWindowAttrib(GlfwApplication::window(), GLFW_VISIBLE)) {
        
        // keep processing messages until window closes.
        if(engine_err == 0 && MxUniverse_Flag(MX_RUNNING)) {
            if(!SUCCEEDED((hr = simulationStep()))) {
                close();
            }
            GlfwApplication::redraw();
        }
    }
    return S_OK;
}

HRESULT MxGlfwApplication::mainLoopIteration(double timeout) {
    HRESULT hr;
    if(engine_err == 0 && MxUniverse_Flag(MX_RUNNING)) {

        // perform a simulation step if universe is in running state
        if(FAILED((hr = simulationStep()))) {
            // window close message
            close();
            
            // process messages until window closes
            while(GlfwApplication::window() &&
                  glfwGetWindowAttrib(GlfwApplication::window(), GLFW_VISIBLE)) {
                GlfwApplication::mainLoopIteration();
            }
            return hr;
        }
    }
    else {
        MxSimulator_Redraw();
    }

    // process messages
    GlfwApplication::mainLoopIteration();
    return S_OK;
}

HRESULT MxGlfwApplication::redraw()
{
    GlfwApplication::redraw();
    return S_OK;
}

void MxGlfwApplication::viewportEvent(ViewportEvent &event)
{
    _ren->viewportEvent(event);
}

void MxGlfwApplication::keyPressEvent(KeyEvent &event)
{
    MxKeyEvent_Invoke(event);
    
    if(event.isAccepted()) {
        return;
    }
    
    bool handled = false;
    switch(event.key()) {
        case Platform::GlfwApplication::KeyEvent::Key::Space: {
            bool current = MxUniverse_Flag(MxUniverse_Flags::MX_RUNNING);
            MxUniverse_SetFlag(MxUniverse_Flags::MX_RUNNING, !current);
            break;
        }
        case Platform::GlfwApplication::KeyEvent::KeyEvent::Key::S: {
            MxUniverse_Step(0, 0);
        }
        default:
            break;
    }
    
    if(handled) {
        event.setAccepted();
    }
    else {
        _ren->keyPressEvent(event);
    }
}

void MxGlfwApplication::mousePressEvent(MouseEvent &event)
{
    _ren->mousePressEvent(event);
}

void MxGlfwApplication::mouseReleaseEvent(MouseEvent &event)
{
    _ren->mouseReleaseEvent(event);
}

void MxGlfwApplication::mouseMoveEvent(MouseMoveEvent &event)
{
    _ren->mouseMoveEvent(event);
}

void MxGlfwApplication::mouseScrollEvent(MouseScrollEvent &event)
{
    _ren->mouseScrollEvent(event);
}

void MxGlfwApplication::exitEvent(ExitEvent &event)
{
    std::cout << MX_FUNCTION << std::endl;

    // stop the window from getting (getting destroyed)
    glfwSetWindowShouldClose(window(), false);


    // "close", actually hide the window.
    close();

    event.setAccepted();
}

HRESULT MxGlfwApplication::destroy()
{
    std::cout << MX_FUNCTION << std::endl;

    GLFWwindow *window = GlfwApplication::window();

    glfwSetWindowShouldClose(window, true);

    return S_OK;

}

HRESULT MxGlfwApplication::close()
{
    std::cout << MX_FUNCTION << std::endl;
    
    glfwHideWindow(window());
    
    return S_OK;
}

int MxGlfwApplication::windowAttribute(MxWindowAttributes attr)
{
    return glfwGetWindowAttrib(window(), attr);
}

HRESULT MxGlfwApplication::setWindowAttribute(MxWindowAttributes attr, int val)
{
    glfwSetWindowAttrib(window(), attr, val);
    MXGLFW_CHECK();
}

#ifdef _WIN32

HRESULT ForceForgoundWindow1(GLFWwindow *wnd) {

    std::fprintf(stderr, "ForceForgoundWindow1 \n");

    HWND window = glfwGetWin32Window(wnd);

    // This implementation registers a hot key (F22) and then
    // triggers the hot key.  When receiving the hot key, we'll
    // be in the foreground and allowed to move the target window
    // into the foreground too.

    //set_window_style(WS_POPUP);
    //Init(NULL, gfx::Rect());

    static const int kHotKeyId = 0x0000baba;
    static const int kHotKeyWaitTimeout = 2000;

    // Store the target window into our USERDATA for use in our
    // HotKey handler.
    RegisterHotKey(window, kHotKeyId, 0, VK_F22);

    // If the calling thread is not yet a UI thread, call PeekMessage
    // to ensure creation of its message queue.
    MSG msg = { 0 };
    PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE);

    // Send the Hotkey.
    INPUT hotkey = { 0 };
    hotkey.type = INPUT_KEYBOARD;
    hotkey.ki.wVk = VK_F22;
    if (1 != SendInput(1, &hotkey, sizeof(hotkey))) {
        std::cerr << "Failed to send input; GetLastError(): " << GetLastError();
        return E_FAIL;
    }

    // There are scenarios where the WM_HOTKEY is not dispatched by the
 // the corresponding foreground thread. To prevent us from indefinitely
 // waiting for the hotkey, we set a timer and exit the loop.
    SetTimer(window, kHotKeyId, kHotKeyWaitTimeout, NULL);

    // Loop until we get the key or the timer fires.
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);

        if (WM_HOTKEY == msg.message)
            break;
        if (WM_TIMER == msg.message) {
            SetForegroundWindow(window);
            break;
        }
    }

    UnregisterHotKey(window, kHotKeyId);
    KillTimer(window, kHotKeyId);

    return S_OK;
}


void ForceForgoundWindow2(GLFWwindow* wnd)
{
    std::fprintf(stderr, "ForceForgoundWindow2 \n");

    HWND hWnd = glfwGetWin32Window(wnd);

    if (!::IsWindow(hWnd)) return;

    //relation time of SetForegroundWindow lock
    DWORD lockTimeOut = 0;
    HWND  hCurrWnd = ::GetForegroundWindow();
    DWORD dwThisTID = ::GetCurrentThreadId(),
        dwCurrTID = ::GetWindowThreadProcessId(hCurrWnd, 0);

    //we need to bypass some limitations from Microsoft :)
    if (dwThisTID != dwCurrTID)
    {
        ::AttachThreadInput(dwThisTID, dwCurrTID, TRUE);

        ::SystemParametersInfo(SPI_GETFOREGROUNDLOCKTIMEOUT, 0, &lockTimeOut, 0);
        ::SystemParametersInfo(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, 0, SPIF_SENDWININICHANGE | SPIF_UPDATEINIFILE);

        ::AllowSetForegroundWindow(ASFW_ANY);
    }

    ::SetForegroundWindow(hWnd);

    if (dwThisTID != dwCurrTID)
    {
        ::SystemParametersInfo(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, (PVOID)lockTimeOut, SPIF_SENDWININICHANGE | SPIF_UPDATEINIFILE);
        ::AttachThreadInput(dwThisTID, dwCurrTID, FALSE);
    }
}

#endif

HRESULT MxGlfwApplication::show()
{
    std::cout << MX_FUNCTION << std::endl;
    
    showWindow();
    
    if (!Mx_IsIpython()) {
        return messageLoop(-1);
    }
    
    return S_OK;
}

HRESULT MxGlfwApplication::showWindow()
{
    std::cout << MX_FUNCTION << std::endl;
    
    glfwShowWindow(window());

#ifdef _WIN32
    if (!Mx_IsIpython()) {
        ForceForgoundWindow1(window());
    }
#endif

    MXGLFW_CHECK();
}
