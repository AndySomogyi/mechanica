/*
 * MxGlfwApplication.h
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 *
 */

#ifndef SRC_MXGLFWAPPLICATION_H_
#define SRC_MXGLFWAPPLICATION_H_

#include <mechanica_private.h>
#include <rendering/MxApplication.h>
#include <MxSimulator.h>
#include <Magnum/Platform/GlfwApplication.h>

#include <rendering/MxGlfwWindow.h>
#include <rendering/MxUniverseRenderer.h>


using namespace Magnum;


class MxGlfwApplication :
        public MxApplication,
        public Magnum::Platform::GlfwApplication {

public:

    typedef Magnum::Platform::GlfwApplication::Arguments Arguments;



    /**
     * creates the app, but does not create the context.
     */
    MxGlfwApplication(const Arguments &args);


    HRESULT createContext(const MxSimulator::Config &conf);

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
     HRESULT pollEvents () override;

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
    HRESULT waitEvents () override;

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

    HRESULT waitEventsTimeout(double  timeout) override;


    /**
     * This function posts an empty event from the current thread
     * to the event queue, causing waitEvents or waitEventsTimeout to return.
     */
    HRESULT postEmptyEvent() override;


    HRESULT setSwapInterval(int si) override;


    void drawEvent() override;


    MxGlfwWindow *getWindow() override;


    MxUniverseRenderer *getRenderer() override;

    HRESULT simulationStep();

    HRESULT run() override;
            
    HRESULT redraw() override;

    MxGlfwWindow *_win;

    int currentStep = 0;

    // TODO implement events and move these to simulator.
    MxUniverseRenderer *_ren;

    Int _substeps = 1;
    bool _pausedSimulation = false;
    bool _mousePressed = false;
    bool _dynamicBoundary = true;

    Float _boundaryOffset = 0.0f; /* For boundary animation */


    /* Timeline to adjust number of simulation steps per frame */
    Timeline _timeline;

    HRESULT mainLoopIteration(double timeout) override;


    void viewportEvent(ViewportEvent& event) override;
    void keyPressEvent(KeyEvent& event) override;
    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;
    void mouseScrollEvent(MouseScrollEvent& event) override;
    void exitEvent(ExitEvent& event) override;

    int windowAttribute(MxWindowAttributes attr) override;

    HRESULT setWindowAttribute(MxWindowAttributes attr, int val) override;

    HRESULT destroy() override;

    HRESULT close() override;

    HRESULT show() override;
            
    HRESULT messageLoop();
            
    HRESULT showWindow();
};


#endif /* SRC_MXGLFWAPPLICATION_H_ */
