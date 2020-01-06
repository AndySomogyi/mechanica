/*
 * mx_ui.h
 *
 *  Created on: Oct 6, 2018
 *      Author: andy
 */

#ifndef INCLUDE_MX_UI_H_
#define INCLUDE_MX_UI_H_

#include <c_port.h>


/**
 * Mechanica needs to communicate regularly with the window system both in order to
 * receive events and to show that the application hasn't locked up. Event
 * processing must be done regularly while you have any windows and is
 * normally done each frame after buffer swapping. Even when you have no
 * windows, event polling needs to be done in order to receive monitor
 * connection events.
 */

/**
 * processes only those events that have already been received and then
 * returns immediately. This is the best choice when rendering continually,
 * like most games do.
 */
CAPI_FUNC(HRESULT) MxUI_PollEvents();


/**
 * If you only need to update the contents of the window when you receive new input,
 * MxUI_WaitEvents is a better choice.
 *
 * It puts the thread to sleep until at least one event has been received and
 * then processes all received events. This saves a great deal of CPU cycles
 * and is useful for, for example, editing tools. There must be at least one
 * Mechanica window for this function to sleep.
 *
 * If you want to wait for events but have UI elements that need periodic updates,
 * set the timeout. It puts the thread to sleep until at least one event
 * has been received, or until the specified number of seconds have elapsed.
 * It then processes any received events.
 *
 * @param timeout: the maximum duration this method should wait before returning.
 */
CAPI_FUNC(HRESULT) MxUI_WaitEvents(double timeout);

/**
 * If the main thread is sleeping in MxUI_WaitEvents, you can wake it from another
 * thread by posting an empty event to the event queue with MxUI_PostEmptyEvent.
 */
CAPI_FUNC(HRESULT) MxUI_PostEmptyEvent();

struct MxGraphicsConfiguration {
    char* appTitle;
    int sampleCount;
    uint32_t version;
    uint32_t flags;
    uint32_t windowFlags;
    uint32_t cursorMode;
    bool srgbCapable;
};

CAPI_FUNC(HRESULT) MxUI_InitializeGraphics(const MxGraphicsConfiguration *conf);


#endif /* INCLUDE_MX_UI_H_ */
