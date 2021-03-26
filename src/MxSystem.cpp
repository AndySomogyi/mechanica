/*
 * MxSystem.cpp
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#include <MxSystem.h>
#include <MxUtil.h>
#include <MxSimulator.h>
#include <rendering/MxWindowlessApplication.h>
#include <rendering/MxGlInfo.h>
#include <rendering/MxWindowless.h>
#include <rendering/MxApplication.h>
#include <rendering/MxUniverseRenderer.h>
#include <rendering/MxGlfwApplication.h>
#include <MxConvert.hpp>


static PyObject *_gl_info(PyObject *mod, PyObject *args, PyObject *kwds) {
    return Mx_GlInfo(args, kwds);
}

static PyObject *_egl_info(PyObject *mod, PyObject *args, PyObject *kwds) {
    return Mx_EglInfo(args, kwds);
}


#if defined(MX_APPLE)

static PyObject *test_headless(PyObject *mod, PyObject *args, PyObject *kwds) {
    return Mx_GlInfo(args, kwds);
}

#elif defined(MX_LINUX)
static PyObject *test_headless(PyObject *mod, PyObject *args, PyObject *kwds) {
    return Mx_GlInfo(args, kwds);
}
#elif defined(MX_WINDOWS)
static PyObject *test_headless(PyObject *mod, PyObject *args, PyObject *kwds) {
    return Mx_GlInfo(args, kwds);
}
#else
#error no windowless application available on this platform
#endif


PyObject *system_camera_move_to(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        MxSimulator *sim = MxSimulator::Get();
        MxUniverseRenderer *rend = sim->app->getRenderer();
        
        const Vector3 eye = mx::arg<Magnum::Vector3>("eye", 0, args, kwargs, rend->defaultEye());
        const Vector3 center = mx::arg<Magnum::Vector3>("center", 1, args, kwargs, rend->defaultCenter());
        const Vector3 up = mx::arg<Magnum::Vector3>("up", 2, args, kwargs, rend->defaultUp());
        
        MxSystem_CameraMoveTo(eye, center, up);

        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_reset(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        MxSystem_CameraReset();
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_rotate_mouse(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector2i mousePos = mx::arg<Magnum::Vector2i>("mouse_pos", 0, args, kwargs);
        MxSystem_CameraRotateMouse(mousePos);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_translate_mouse(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector2i mousePos = mx::arg<Magnum::Vector2i>("mouse_pos", 0, args, kwargs);
        MxSystem_CameraTranslateMouse(mousePos);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_init_mouse(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector2i mousePos = mx::arg<Magnum::Vector2i>("mouse_pos", 0, args, kwargs);
        MxSystem_CameraInitMouse(mousePos);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_translate_by(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector2 trans = mx::arg<Magnum::Vector2>("translate", 0, args, kwargs);
        MxSystem_CameraTranslateDelta(trans);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_zoom_by(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        float delta = mx::arg<float>("delta", 0, args, kwargs);
        MxSystem_CameraZoomBy(delta);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_zoom_to(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        float delta = mx::arg<float>("distance", 0, args, kwargs);
        MxSystem_CameraZoomTo(delta);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_rotate_to_axis(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector3 axis = mx::arg<Magnum::Vector3>("axis", 0, args, kwargs);
        float distance = mx::arg<float>("distance", 1, args, kwargs);
        MxSystem_CameraRotateToAxis(axis,  distance);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_rotate_to_euler_angle(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector3 angles = mx::arg<Magnum::Vector3>("angles", 0, args, kwargs);
        MxSystem_CameraRotateToEulerAngle(angles);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_camera_rotate_by_euler_angle(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector3 angles = mx::arg<Magnum::Vector3>("angles", 0, args, kwargs);
        MxSystem_CameraRotateByEulerAngle(angles);
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *system_view_reshape(PyObject *self, PyObject *args, PyObject *kwargs)
{
    try {
        Magnum::Vector2i windowSize = mx::arg<Magnum::Vector2i>("window_size", 0, args, kwargs);
        MxSystem_ViewReshape(windowSize );
        Py_RETURN_NONE;
    }
    catch(const std::exception &e) {
        C_RETURN_EXP(e);
    }
}

PyObject *is_terminal_interactive(PyObject *o) {
    if(C_TerminalInteractiveShell()) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

PyObject *is_jupyter_notebook(PyObject *o) {
    if(C_ZMQInteractiveShell()) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}


PyObject *MxSystem_JWidget_Init(PyObject *args, PyObject *kwargs) {
    
    PyObject* moduleString = PyUnicode_FromString((char*)"mechanica.jwidget");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        C_ERR(E_FAIL, "could not import mechanica.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* init = PyObject_GetAttrString(module,(char*)"init");
    
    if(!init) {
        C_ERR(E_FAIL, "mechanica.jwidget package does not have an init function");
        return NULL;
    }

    PyObject* result = PyObject_Call(init, args, kwargs);
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(init);
    
    if(!result) {
        Log(LOG_ERROR) << "error calling mechanica.jwidget.init: " << carbon::pyerror_str();
    }
    
    return result;
}

PyObject *MxSystem_JWidget_Run(PyObject *args, PyObject *kwargs) {
    PyObject* moduleString = PyUnicode_FromString((char*)"mechanica.jwidget");
    
    if(!moduleString) {
        return NULL;
    }
    
    #if defined(__has_feature)
    #  if __has_feature(thread_sanitizer)
        std::cout << "thread sanitizer, returning NULL" << std::endl;
        return NULL;
    #  endif
    #endif
    
    PyObject* module = PyImport_Import(moduleString);
    if(!module) {
        C_ERR(E_FAIL, "could not import mechanica.jwidget package");
        return NULL;
    }
    
    // Then getting a reference to your function :

    PyObject* run = PyObject_GetAttrString(module,(char*)"run");
    
    if(!run) {
        C_ERR(E_FAIL, "mechanica.jwidget package does not have an run function");
        return NULL;
    }

    PyObject* result = PyObject_Call(run, args, kwargs);
    
    Py_DECREF(moduleString);
    Py_DECREF(module);
    Py_DECREF(run);
    
    return result;
    
}

static PyMethodDef system_methods[] = {
    { "cpu_info", (PyCFunction)MxInstructionSetFeatruesDict, METH_NOARGS, NULL },
    //{ "compile_flags", (PyCFunction)MxCompileFlagsDict, METH_NOARGS, NULL },
    { "gl_info", (PyCFunction)_gl_info, METH_VARARGS | METH_KEYWORDS, NULL },
    { "egl_info", (PyCFunction)_egl_info, METH_VARARGS | METH_KEYWORDS, NULL },
    { "test_headless", (PyCFunction)test_headless, METH_VARARGS | METH_KEYWORDS, NULL },
    { "test_image", (PyCFunction)MxTestImage, METH_VARARGS | METH_KEYWORDS, NULL },
    { "image_data", (PyCFunction)MxFramebufferImageData, METH_VARARGS | METH_KEYWORDS, NULL },
    { "context_has_current", (PyCFunction)MxSystem_ContextHasCurrent, METH_NOARGS, NULL },
    { "context_make_current", (PyCFunction)MxSystem_ContextMakeCurrent, METH_NOARGS, NULL },
    { "context_release", (PyCFunction)MxSystem_ContextRelease,  METH_NOARGS, NULL },
    { "camera_move_to", (PyCFunction)system_camera_move_to, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_reset", (PyCFunction)system_camera_reset, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_rotate_mouse", (PyCFunction)system_camera_rotate_mouse, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_translate_mouse", (PyCFunction)system_camera_translate_mouse, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_init_mouse", (PyCFunction)system_camera_init_mouse, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_translate_by", (PyCFunction)system_camera_translate_by, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_zoom_by", (PyCFunction)system_camera_zoom_by, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_zoom_to", (PyCFunction)system_camera_zoom_to, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_rotate_to_axis", (PyCFunction)system_camera_rotate_to_axis, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_rotate_to_euler_angle", (PyCFunction)system_camera_rotate_to_euler_angle, METH_VARARGS | METH_KEYWORDS, NULL},
    { "camera_rotate_by_euler_angle", (PyCFunction)system_camera_rotate_by_euler_angle, METH_VARARGS | METH_KEYWORDS, NULL},
    { "view_reshape", (PyCFunction)system_view_reshape, METH_VARARGS | METH_KEYWORDS, NULL},
    { "is_terminal_interactive", (PyCFunction)is_terminal_interactive, METH_VARARGS | METH_KEYWORDS, NULL},
    { "is_jupyter_notebook", (PyCFunction)is_jupyter_notebook, METH_VARARGS | METH_KEYWORDS, NULL},
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef system_def = {
    PyModuleDef_HEAD_INIT,
    "system",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
               or -1 if the module keeps state in global variables. */
    system_methods
};


void MxSystem_CameraMoveTo(const Magnum::Vector3 &eye,
        const Magnum::Vector3 &viewCenter, const Magnum::Vector3 &upDir)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->setViewParameters(eye, viewCenter, upDir);
}

void MxSystem_CameraReset()
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
        
    ab->reset();
}

void MxSystem_CameraRotateMouse(const Magnum::Vector2i &mousePos)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->rotate(mousePos);
}

void MxSystem_CameraTranslateMouse(const Magnum::Vector2i &mousePos)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->translate(mousePos);
}

void MxSystem_CameraTranslateDelta(const Magnum::Vector2 &translationNDC)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->translateDelta(translationNDC);
}

void MxSystem_CameraZoomBy(float delta)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->zoom(delta);
}

void MxSystem_CameraZoomTo(float delta)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->zoomTo(delta);
}

void MxSystem_CameraRotateToAxis(const Magnum::Vector3 &axis, float distance)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->rotateToAxis(axis, distance);
}

void MxSystem_CameraRotateToEulerAngle(const Magnum::Vector3 &angles)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->rotateToEulerAngles(angles);
}

void MxSystem_CameraRotateByEulerAngle(const Magnum::Vector3 &anglesDelta)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->rotateByEulerAngles(anglesDelta);
    
    MxSimulator_Redraw();
}

void MxSystem_ViewReshape(const Magnum::Vector2i &windowSize)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->reshape(windowSize);
}



void MxSystem_CameraInitMouse(const Magnum::Vector2i& mousePos)
{
    MxSimulator *sim = MxSimulator::Get();
    
    MxUniverseRenderer *renderer = sim->app->getRenderer();
    
    Magnum::Mechanica::ArcBall *ab = renderer->_arcball;
    
    ab->initTransformation(mousePos);
    
    MxSimulator_Redraw();
}

HRESULT _MxSystem_init(PyObject* m) {

    PyObject *system_module = PyModule_Create(&system_def);

    if(!system_module) {
        return c_error(E_FAIL, "could not create system module");
    }

    if(PyModule_AddObject(m, "system", system_module) != 0) {
        return c_error(E_FAIL, "could not add system module to mechanica");
    }

    //if(PyModule_AddObject(m, "version", version_create()) != 0) {
    //    std::cout << "error creating version info module" << std::endl;
    //}

    return S_OK;
}
