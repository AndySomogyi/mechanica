/*
 * MxApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <MxApplication.h>
#include <MxWindowlessApplication.h>

#include <iostream>

static MxApplication* app = nullptr;
static MxObject *obj = nullptr;



/**
 * tp_alloc(type) to allocate storage
 * tp_new(type, args) to create blank object
 * tp_init(obj, args) to initialize object
 */

static int MxApplication_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    std::cout << MX_FUNCTION << std::endl;
    return 0;
}

static PyTypeObject ApplicationType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mechanica.Application",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(MxApplication),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = MxApplication_init,
};

PyTypeObject *MxApplication_Type = &ApplicationType;

static PyModuleDef custommodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "custom",
    .m_doc = "Example module that creates an extension type.",
    .m_size = -1,
};

HRESULT MxApplication_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)MxApplication_Type) < 0)
        return E_FAIL;



    Py_INCREF(MxApplication_Type);
    PyModule_AddObject(m, "Application", (PyObject *) MxApplication_Type);

    return 0;
}

HRESULT MxApplication::create(int argc, char** argv, const Configuration& conf)
{
    if(!app) {
        app = new MxWindowlessApplication(argc, argv, conf);

        std::cout << "created new app: " << std::hex << app << std::endl;
    }

    obj = new MxObject();


    return S_OK;
}

HRESULT MxApplication::destroy()
{
    std::cout << "destroying app: " << std::hex << app << std::endl;

    delete obj;
    delete app;
    app = nullptr;
    return S_OK;
}

MxApplication::MxApplication()
{
}

MxApplication* MxApplication::get()
{
    return app;
}
