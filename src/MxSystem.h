/*
 * MxSystem.h
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#ifndef SRC_MXSYSTEM_H_
#define SRC_MXSYSTEM_H_

#include "mechanica_private.h"
#include "MxModel.h"
#include "MxPropagator.h"
#include "MxController.h"
#include "MxView.h"


/* Set the camera view parameters: eye position, view center, up
   direction */
void MxSystem_CameraMoveTo(const Magnum::Vector3& eye, const Magnum::Vector3& viewCenter,
                        const Magnum::Vector3& upDir);

/* */
void MxSystem_CameraReset();

/* Rotate the camera from the previous (screen) mouse position to the
   current (screen) position */
void MxSystem_CameraRotateMouse(const Magnum::Vector2i& mousePos);


/* Rotate the camera from the previous (screen) mouse position to the
   current (screen) position */
void MxSystem_CameraInitMouse(const Magnum::Vector2i& mousePos);

/* Translate the camera from the previous (screen) mouse position to
   the current (screen) mouse position */
void MxSystem_CameraTranslateMouse(const Magnum::Vector2i& mousePos);

/* Translate the camera by the delta amount of (NDC) mouse position.
   Note that NDC position must be in [-1, -1] to [1, 1]. */
void MxSystem_CameraTranslateDelta(const Magnum::Vector2& translationNDC);

/* Zoom the camera (positive delta = zoom in, negative = zoom out) */
void MxSystem_CameraZoomBy(float delta);

/* Zoom the camera (positive delta = zoom in, negative = zoom out) */
void MxSystem_CameraZoomTo(float distance);

/*
 * Set the camera view parameters: eye position, view center, up
 * direction, only rotates the view to the given eye position.
 */
void MxSystem_CameraRotateToAxis(const Magnum::Vector3& axis, float distance);


void MxSystem_CameraRotateToEulerAngle(const Magnum::Vector3& angles);

void MxSystem_CameraRotateByEulerAngle(const Magnum::Vector3& anglesDelta);


/* Update screen size after the window has been resized */
void MxSystem_ViewReshape(const Magnum::Vector2i& windowSize);


PyObject *MxSystem_JWidget_Init(PyObject *args, PyObject *kwargs);

PyObject *MxSystem_JWidget_Run(PyObject *args, PyObject *kwargs);

HRESULT MxLoggerCallback(CLogEvent, std::ostream *);

HRESULT _MxSystem_init(PyObject *m);

#endif /* SRC_MXSYSTEM_H_ */
