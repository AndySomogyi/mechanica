/*
 * MxApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <MxApplication.h>
#include <MxWindowlessApplication.h>


#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/Shaders/VertexColor.h>
#include <MagnumPlugins/TgaImageConverter/TgaImageConverter.h>

#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>

#include <Corrade/Utility/Directory.h>
#include <MxImageConverters.h>



using namespace Magnum;
using namespace Magnum::Trade;

#include <iostream>

static MxApplication* gApp = nullptr;
static MxObject *obj = nullptr;


struct PyApplication : _object {
};


/**
 * tp_alloc(type) to allocate storage
 * tp_new(type, args) to create blank object
 * tp_init(obj, args) to initialize object
 */



static void _dealloc(MxApplication *app) {
    std::cout << MX_FUNCTION << std::endl;
    assert(app == gApp);
    delete app->impl;
    PyObject_Del(app);
    gApp = NULL;
}

PyObject *_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    int argc = 1;
    char* argv[] = {"foo"};
    MxApplicationConfig conf = {};

    return MxApplication_New(argc, argv, &conf);
}

static PyObject* _testImage(PyObject* self, PyObject* args) {

    using namespace Math::Literals;

    struct TriangleVertex {
        Vector2 position;
        Color3 color;
    };
    const TriangleVertex data[]{
        {{-0.5f, -0.5f}, 0xff0000_rgbf},    /* Left vertex, red color */
        {{ 0.5f, -0.5f}, 0x00ff00_rgbf},    /* Right vertex, green color */
        {{ 0.0f,  0.5f}, 0x0000ff_rgbf}     /* Top vertex, blue color */
    };

    GL::Buffer buffer;
    buffer.setData(data);

    GL::Mesh mesh;
    mesh.setCount(3)
         .addVertexBuffer(std::move(buffer), 0,
            Shaders::VertexColor2D::Position{},
            Shaders::VertexColor2D::Color3{});

    GL::Renderbuffer renderbuffer;
    renderbuffer.setStorage(GL::RenderbufferFormat::RGBA8, {640, 480});
    GL::Framebuffer framebuffer{{{}, {640, 480}}};
    framebuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderbuffer)
        .clear(GL::FramebufferClear::Color)
        .bind();

    Shaders::VertexColor2D shader;
    mesh.draw(shader);



    const GL::PixelFormat format = framebuffer.implementationColorReadFormat();
    Image2D image = framebuffer.read(framebuffer.viewport(), PixelFormat::RGBA8Unorm);

    TgaImageConverter conv;

    conv.exportToFile(image, "triangle.tga");

    auto jpegData = convertImageDataToJpeg(image);


    /* Open file */
    if(!Utility::Directory::write("triangle.jpg", jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return NULL;
    }


    return PyBytes_FromStringAndSize(jpegData.data(), jpegData.size());
}

static PyMethodDef methods[] = {
    {"testImage", (PyCFunction)_testImage, METH_VARARGS,  "make a test image" },
    {NULL}  /* Sentinel */
};



static PyTypeObject ApplicationType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "mechanica.Application",
    .tp_basicsize = sizeof(MxApplication),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)_dealloc,
    .tp_print = 0, 
    .tp_getattr = 0, 
    .tp_setattr = 0, 
    .tp_as_async = 0, 
    .tp_repr = 0, 
    .tp_as_number = 0, 
    .tp_as_sequence = 0, 
    .tp_as_mapping = 0, 
    .tp_hash = 0, 
    .tp_call = 0, 
    .tp_str = 0, 
    .tp_getattro = 0, 
    .tp_setattro = 0, 
    .tp_as_buffer = 0, 
    .tp_flags = Py_TPFLAGS_DEFAULT, 
    .tp_doc = 0, 
    .tp_traverse = 0, 
    .tp_clear = 0, 
    .tp_richcompare = 0, 
    .tp_weaklistoffset = 0, 
    .tp_iter = 0, 
    .tp_iternext = 0, 
    .tp_methods = methods,
    .tp_members = 0, 
    .tp_getset = 0, 
    .tp_base = 0, 
    .tp_dict = 0, 
    .tp_descr_get = 0, 
    .tp_descr_set = 0, 
    .tp_dictoffset = 0, 
    .tp_init = 0,
    .tp_alloc = 0, 
    .tp_new = _new,
    .tp_free = 0,
    .tp_is_gc = 0, 
    .tp_bases = 0, 
    .tp_mro = 0, 
    .tp_cache = 0, 
    .tp_subclasses = 0, 
    .tp_weaklist = 0, 
    .tp_del = 0, 
    .tp_version_tag = 0, 
    .tp_finalize = 0, 
};

/*
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
*/

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
    std::cout << MX_FUNCTION << std::endl;

    if(!gApp) {
        gApp = new MxApplication();
        gApp->impl = new MxWindowlessApplication(argc, argv, conf);

        std::cout << "created new app: " << std::hex << gApp << std::endl;
    }

    obj = new MxObject();


    return S_OK;
}

HRESULT MxApplication::destroy()
{
    std::cout << "destroying app: " << std::hex << gApp << std::endl;

    delete obj;

    gApp = nullptr;
    return S_OK;
}



MxApplication* MxApplication::get()
{
    return gApp;
}

PyObject* MxApplication_New(int argc, char** argv,
        const MxApplicationConfig* conf)
{
    if(!gApp) {
        
        MxApplication *o = (MxApplication *) PyObject_MALLOC(sizeof(MxApplication));
        PyObject_INIT( o, MxApplication_Type );

        gApp = o;

        gApp->impl = new MxWindowlessApplication(argc, argv, *conf);

        std::cout << "created new app: " << std::hex << gApp << std::endl;

    }


    Py_INCREF(gApp);

    std::cout << "returning app, ref count: " << gApp->ob_refcnt << std::endl;
    return (PyObject*)gApp;
}
