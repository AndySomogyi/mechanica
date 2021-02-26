/*
 * MxSystem.cpp
 *
 *  Created on: Apr 2, 2017
 *      Author: andy
 */

#include <MxSystem.h>
#include <MxUtil.h>
#include <MxSimulator.h>
#include <rendering/MxGlInfo.h>
#include <rendering/MxWindowless.h>
#include <rendering/MxApplication.h>


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
    { "camera_rotate", (PyCFunction)MxSystem_CameraRotate,  METH_VARARGS | METH_KEYWORDS, NULL },

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
