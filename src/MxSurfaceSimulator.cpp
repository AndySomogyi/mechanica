/*
 * MxSurfaceSimulator.cpp
 *
 *  Created on: Mar 28, 2019
 *      Author: andy
 */

#include <MxSurfaceSimulator.h>
#include "MeshOperations.h"
#include "Magnum/GL/Version.h"
#include "Magnum/Platform/GLContext.h"

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Trade/MeshData3D.h>

#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Corrade/Utility/Directory.h>
#include <MxImageConverters.h>



#include <memory>
#include <iostream>

using namespace Math::Literals;



/**
 * tp_alloc(type) to allocate storage
 * tp_new(type, args) to create blank object
 * tp_init(obj, args) to initialize object
 */

static int _init(PyObject *obj, PyObject *args, PyObject *kwds)
{
    std::cout << MX_FUNCTION << std::endl;
    std::cout << "count: " << obj->ob_refcnt << std::endl;
    std::cout << "ob_type: " << obj->ob_type << std::endl;
    std::cout << "name: " << obj->ob_type->tp_name << std::endl;
    std::cout << "size: " << obj->ob_type->tp_basicsize << std::endl;

    const std::string dirName = MX_MODEL_DIR;
    std::string modelPath = dirName  + "/hex_cylinder.1.obj";

    MxSurfaceSimulator_Config simConf = {{900,500}, modelPath.c_str()};
    

    MxSurfaceSimulator *self = new(obj) MxSurfaceSimulator(simConf);


    std::cout << MX_FUNCTION << "completed initialization" << std::endl;
    return 0;
}

static PyObject* _name(MxSurfaceSimulator* self)
{
    return PyUnicode_FromFormat("%s %s", "foo", "bar");
}

static PyObject* _imageData(MxSurfaceSimulator* self, PyObject* args) {

    const GL::PixelFormat format = self->frameBuffer.implementationColorReadFormat();
    Image2D image = self->frameBuffer.read(self->frameBuffer.viewport(), PixelFormat::RGBA8Unorm);

    auto jpegData = convertImageDataToJpeg(image);

    /* Open file */
    if(!Utility::Directory::write("SurfaceSimulator.jpg", jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return NULL;
    }

    return PyBytes_FromStringAndSize(jpegData.data(), jpegData.size());
}



static PyMethodDef methods[] = {
    {"name", (PyCFunction)_name, METH_NOARGS,  "Return the name, combining the first and last name"},
    {"imageData", (PyCFunction)_imageData, METH_VARARGS,  "Return the name, combining the first and last name"},
    {NULL}  /* Sentinel */
};


static PyTypeObject SurfaceSimulatorType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "mechanica.SurfaceSimulator",
    .tp_basicsize = sizeof(MxSurfaceSimulator),
    .tp_itemsize = 0,
    .tp_dealloc = 0,
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
    .tp_init = (initproc)_init,
    .tp_alloc = 0, 
    .tp_new = PyType_GenericNew, 
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
static PyTypeObject SurfaceSimulatorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mechanica.SurfaceSimulator",
    .tp_doc = "Custom objects",
    .tp_basicsize = sizeof(MxSurfaceSimulator),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = init,
    .tp_methods = methods
};
*/

PyTypeObject *MxSurfaceSimuator_Type = &SurfaceSimulatorType;

MxSurfaceSimulator* MxSurfaceSimulator_New(
        const MxSurfaceSimulator_Config* conf)
{
    PyObject *obj = PyType_GenericNew(MxSurfaceSimuator_Type, nullptr, nullptr);
    
    std::cout << "count: " << obj->ob_refcnt << std::endl;
    std::cout << "ob_type: " << obj->ob_type << std::endl;
    std::cout << "name: " << obj->ob_type->tp_name << std::endl;
    std::cout << "size: " << obj->ob_type->tp_basicsize << std::endl;


    MxSurfaceSimulator *result = new(obj) MxSurfaceSimulator(*conf);
    
    std::cout << "after ctor:" << std::endl;
    std::cout << "count: " << obj->ob_refcnt << std::endl;
    std::cout << "ob_type: " << obj->ob_type << std::endl;
    std::cout << "name: " << obj->ob_type->tp_name << std::endl;
    std::cout << "size: " << obj->ob_type->tp_basicsize << std::endl;

    return result;
}

HRESULT MxSurfaceSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;


    if (PyType_Ready((PyTypeObject *)MxSurfaceSimuator_Type) < 0)
        return E_FAIL;


    Py_INCREF(MxSurfaceSimuator_Type);
    PyModule_AddObject(m, "SurfaceSimulator", (PyObject *) MxSurfaceSimuator_Type);

    return 0;
}

MxSurfaceSimulator::MxSurfaceSimulator(const Configuration& config) :
        frameBuffer{Magnum::NoCreate}
{
    std::cout << MX_FUNCTION << std::endl;

    createContext(config);

    // need to enabler depth testing. The graphics processor can draw each facet in any order it wants.
    // Depth testing makes sure that front facing facts are drawn after back ones, so that back facets
    // don't cover up front ones.
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);

    // don't draw facets that face away from us. We have A LOT of these INSIDE cells, no need to
    // draw them.
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable( GL_BLEND );

    renderer = new MxMeshRenderer{MxMeshRenderer::Flag::Wireframe};

    GL::Renderer::setClearColor(Color4{1.0f, 1.0f, 1.0f, 1.0f});

    renderBuffer.setStorage(GL::RenderbufferFormat::RGBA8, {config.frameBufferSize[0], config.frameBufferSize[1]});

    frameBuffer = GL::Framebuffer{{{0,0}, {config.frameBufferSize[0], config.frameBufferSize[1]}}};

    frameBuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderBuffer)
            .clear(GL::FramebufferClear::Color)
            .bind();
    
    loadModel(config.modelPath);
}


HRESULT MxSurfaceSimulator::createContext(const Configuration& configuration) {

    //CORRADE_ASSERT(context->version() ==
    //        GL::Version::None,
    //        "Platform::GlfwApplication::tryCreateContext(): context already created",
    //        false);

    /* Window flags */

    if(MxApplication::get() == nullptr) {
        std::cout << "WTF, application does not exist... \n";


    }
    
    assert(Magnum::GL::Context::hasCurrent() && "must have context, should be created by application");

    return S_OK;
}

void MxSurfaceSimulator::loadModel(const char* fileName)
{
    std::cout << MX_FUNCTION << ", fileName: " << fileName << std::endl;

    delete model;
    delete propagator;

    model = new MxCylinderModel{};

    propagator = new LangevinPropagator{};

    VERIFY(MxBind_PropagatorModel(propagator, model));

    VERIFY(model->loadModel(fileName));

    renderer->setMesh(model->mesh);

    draw();
}

void MxSurfaceSimulator::draw() {
    
    frameBuffer.bind();

    Vector3 min, max;
    std::tie(min, max) = model->mesh->extents();

    center = (max + min)/2;

    frameBuffer.clear(GL::FramebufferClear::Color|GL::FramebufferClear::Depth);

    renderer->setViewportSize(Vector2{frameBuffer.viewport().size()});

    projection = Matrix4::perspectiveProjection(35.0_degf,
        Vector2{frameBuffer.viewport().size()}.aspectRatio(),
        0.01f, 100.0f
        );

    renderer->setProjectionMatrix(projection);

    //rotation = arcBall.rotation();
    
    rotation = Matrix4::rotationZ(-1.4_radf);
    
    rotation = rotation * Matrix4::rotationX(0.5_radf);

    Matrix4 mat = Matrix4::translation(centerShift) * rotation * Matrix4::translation(-center) ;

    renderer->setViewMatrix(mat);

    renderer->setColor(Color4::yellow());

    renderer->setWireframeColor(Color4{0., 0., 0.});

    renderer->setWireframeWidth(2.0);

    Debug{} << "viewport size: " << frameBuffer.viewport().size();
    Debug{} << "center: " << center;
    Debug{} << "centerShift: " << centerShift;
    Debug{} << "projection: " << projection;
    Debug{} << "view matrix: " << mat;

    renderer->draw();
}

PyObject* MxSurfaceSimulator_ImageData(MxSurfaceSimulator* self,
        const char* path)
{

    const GL::PixelFormat format = self->frameBuffer.implementationColorReadFormat();
    Image2D image = self->frameBuffer.read(self->frameBuffer.viewport(), PixelFormat::RGBA8Unorm);


    auto jpegData = convertImageDataToJpeg(image);


    /* Open file */
    if(!Utility::Directory::write(path, jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return NULL;
    }


    return PyBytes_FromStringAndSize(jpegData.data(), jpegData.size());
}
