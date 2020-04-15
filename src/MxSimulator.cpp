/*
 * MxSimulator.cpp
 *
 *  Created on: Feb 1, 2017
 *      Author: andy
 */

#include <MxSimulator.h>
#include <MxUI.h>
#include <MxTestView.h>

#include <Magnum/GL/Context.h>

#if defined(MX_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
#elif defined(MX_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"
#elif defined(MX_WINDOWS)
    #include "Magnum/Platform/WindowlessWglApplication.h"
#else
    #error no windowless application available on this platform
#endif


#include "MxGlfwApplication.h"

#include <map>

#include <Corrade/Utility/Arguments.h>
#include <Corrade/Utility/Debug.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/String.h>

#include "Magnum/GL/AbstractShaderProgram.h"
#include "Magnum/GL/Buffer.h"
#if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
#include "Magnum/GL/BufferTexture.h"
#endif
#include "Magnum/GL/Context.h"
#include "Magnum/GL/CubeMapTexture.h"
#if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
#include "Magnum/GL/CubeMapTextureArray.h"
#endif
#ifndef MAGNUM_TARGET_WEBGL
#include "Magnum/GL/DebugOutput.h"
#endif
#include "Magnum/GL/Extensions.h"
#include "Magnum/GL/Framebuffer.h"
#include "Magnum/GL/Mesh.h"
#if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
#include "Magnum/GL/MultisampleTexture.h"
#endif
#ifndef MAGNUM_TARGET_GLES
#include "Magnum/GL/RectangleTexture.h"
#endif
#include "Magnum/GL/Renderer.h"
#include "Magnum/GL/Renderbuffer.h"
#include "Magnum/GL/Shader.h"
#include "Magnum/GL/Texture.h"
#ifndef MAGNUM_TARGET_GLES2
#include "Magnum/GL/TextureArray.h"
#include "Magnum/GL/TransformFeedback.h"
#endif


#include <sstream>
using namespace Magnum::Platform;



class MxWindowlessApplication : public Platform::WindowlessApplication {
public:
    explicit MxWindowlessApplication(const Arguments& arguments) :
    Platform::WindowlessApplication{arguments, NoCreate} {

    }

    int exec() override { return 0; }

    bool tryCreateContext(const Configuration& configuration) {
        return Platform::WindowlessApplication::tryCreateContext(configuration);
    }
};

static std::string gl_info(const Magnum::Utility::Arguments &args);


static PyObject *not_initialized_error();



MxSimulator* Mx_Simulator = NULL;

// (5) Initializer list constructor
const std::map<std::string, int> configItemMap {
    {"none", MXSIMULATOR_NONE},
    {"windowless", MXSIMULATOR_WINDOWLESS},
    {"glfw", MXSIMULATOR_GLFW}
};


/**
 * tp_alloc(type) to allocate storage
 * tp_new(type, args) to create blank object
 * tp_init(obj, args) to initialize object
 */

static int init(PyObject *self, PyObject *args, PyObject *kwds)
{
    std::cout << MX_FUNCTION << std::endl;

    MxSimulator *s = new (self) MxSimulator();
    return 0;
}

static PyObject *Noddy_name(MxSimulator* self)
{
    return PyUnicode_FromFormat("%s %s", "foo", "bar");
}




static PyObject *simulator_gl_info(PyTypeObject *type, PyObject *args, PyObject *kwds) {

    Magnum::Utility::Arguments arg;

    arg.addBooleanOption('s', "short").setHelp("short", "display just essential info and exit")
        .addBooleanOption("extension-strings").setHelp("extension-strings", "list all extension strings provided by the driver (implies --short)")
        .addBooleanOption("all-extensions").setHelp("all-extensions", "display extensions also for fully supported versions")
        .addBooleanOption("limits").setHelp("limits", "display also limits and implementation-defined values")
        .addSkippedPrefix("magnum", "engine-specific options")
        .setGlobalHelp("Displays information about Magnum engine and OpenGL capabilities.");

    int argc = 0;
    char **argv = nullptr;

    arg.parse(argc, argv);

    std::string str = gl_info(arg);

    return PyUnicode_FromString(str.c_str());
}




static PyObject *simulator_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    std::cout << MX_FUNCTION << std::endl;
    
    if(Mx_Simulator) {
        Py_INCREF(Mx_Simulator);
        return Mx_Simulator;
    }

    int argc = 0;
    char** argv = NULL;
    Platform::WindowlessApplication::Arguments margs(argc, argv);

    MxWindowlessApplication *app = new MxWindowlessApplication(margs);
    
    bool result = app->tryCreateContext({});
    
    if(!result) {
        std::cout << "could not create context..." << std::endl;
        delete app;
        
        Py_RETURN_NONE;
    }
    


    Mx_Simulator = (MxSimulator *) type->tp_alloc(type, 0);

    Mx_Simulator->applicaiton = app;
    Mx_Simulator->kind = MXSIMULATOR_WINDOWLESS;

    return (PyObject *) Mx_Simulator;
}





#if 0
PyTypeObject THPLegacyVariableType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._LegacyVariableBase",        /* tp_name */
    0,                                     /* tp_basicsize */
    0,                                     /* tp_itemsize */
    0,                                     /* tp_dealloc */
    0,                                     /* tp_print */
    0,                                     /* tp_getattr */
    0,                                     /* tp_setattr */
    0,                                     /* tp_reserved */
    0,                                     /* tp_repr */
    0,                                     /* tp_as_number */
    0,                                     /* tp_as_sequence */
    0,                                     /* tp_as_mapping */
    0,                                     /* tp_hash  */
    0,                                     /* tp_call */
    0,                                     /* tp_str */
    0,                                     /* tp_getattro */
    0,                                     /* tp_setattro */
    0,                                     /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr,                               /* tp_doc */
    0,                                     /* tp_traverse */
    0,                                     /* tp_clear */
    0,                                     /* tp_richcompare */
    0,                                     /* tp_weaklistoffset */
    0,                                     /* tp_iter */
    0,                                     /* tp_iternext */
    0,                                     /* tp_methods */
    0,                                     /* tp_members */
    0,                                     /* tp_getset */
    0,                                     /* tp_base */
    0,                                     /* tp_dict */
    0,                                     /* tp_descr_get */
    0,                                     /* tp_descr_set */
    0,                                     /* tp_dictoffset */
    0,                                     /* tp_init */
    0,                                     /* tp_alloc */
    0                      /* tp_new */
};
#endif


#define MX_CLASS METH_CLASS | METH_VARARGS | METH_KEYWORDS




static PyMethodDef methods[] = {
        { "pollEvents", (PyCFunction)MxPyUI_PollEvents, MX_CLASS, NULL },
        { "gl_info", (PyCFunction)simulator_gl_info, MX_CLASS, NULL },
        { "waitEvents", (PyCFunction)MxPyUI_WaitEvents, MX_CLASS, NULL },
        { "postEmptyEvent", (PyCFunction)MxPyUI_PostEmptyEvent, MX_CLASS, NULL },
        { "initializeGraphics", (PyCFunction)MxPyUI_InitializeGraphics, MX_CLASS, NULL },
        { "createTestWindow", (PyCFunction)MxPyUI_CreateTestWindow, MX_CLASS, NULL },
        { "testWin", (PyCFunction)PyTestWin, MX_CLASS, NULL },
        { "destroyTestWindow", (PyCFunction)MxPyUI_DestroyTestWindow, MX_CLASS, NULL },
        { NULL, NULL, 0, NULL }
};


static PyTypeObject MxSimulator_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    .tp_name = "mechanica.Simulator",
    .tp_basicsize = sizeof(MxSimulator),
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
    .tp_init = (initproc)0,
    .tp_alloc = 0, 
    .tp_new = simulator_new,
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



HRESULT MxSimulator_init(PyObject* m) {

    std::cout << MX_FUNCTION << std::endl;



    if (PyType_Ready((PyTypeObject *)&MxSimulator_Type) < 0)
        return E_FAIL;



    Py_INCREF(&MxSimulator_Type);
    PyModule_AddObject(m, "Simulator", (PyObject *) &MxSimulator_Type);

    return 0;
}

CAPI_FUNC(MxSimulator*) MxSimulator_New(MxSimulator_ConfigurationItem *items)
{
}

CAPI_FUNC(MxSimulator*) MxSimulator_Get()
{
}


std::string gl_info(const Magnum::Utility::Arguments &args) {



    std::stringstream os;


    os << "";
    os << "  +---------------------------------------------------------+";
    os << "  |   Information about Magnum engine OpenGL capabilities   |";
    os << "  +---------------------------------------------------------+";
    os << "";

    #ifdef MAGNUM_WINDOWLESSEGLAPPLICATION_MAIN
    os << "Used application: Platform::WindowlessEglApplication";
    #elif defined(MAGNUM_WINDOWLESSIOSAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessIosApplication";
    #elif defined(MAGNUM_WINDOWLESSCGLAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessCglApplication";
    #elif defined(MAGNUM_WINDOWLESSGLXAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessGlxApplication";
    #elif defined(MAGNUM_WINDOWLESSWGLAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessWglApplication";
    #elif defined(MAGNUM_WINDOWLESSWINDOWSEGLAPPLICATION_MAIN)
    os << "Used application: Platform::WindowlessWindowsEglApplication";
    #else
    #error no windowless application available on this platform
    #endif
    os << "Compilation flags:";
    #ifdef CORRADE_BUILD_DEPRECATED
    os << "    CORRADE_BUILD_DEPRECATED";
    #endif
    #ifdef CORRADE_BUILD_STATIC
    os << "    CORRADE_BUILD_STATIC";
    #endif
    #ifdef CORRADE_BUILD_MULTITHREADED
    os << "    CORRADE_BUILD_MULTITHREADED";
    #endif
    #ifdef CORRADE_TARGET_UNIX
    os << "    CORRADE_TARGET_UNIX";
    #endif
    #ifdef CORRADE_TARGET_APPLE
    os << "    CORRADE_TARGET_APPLE";
    #endif
    #ifdef CORRADE_TARGET_IOS
    os << "    CORRADE_TARGET_IOS";
    #endif
    #ifdef CORRADE_TARGET_WINDOWS
    os << "    CORRADE_TARGET_WINDOWS";
    #endif
    #ifdef CORRADE_TARGET_WINDOWS_RT
    os << "    CORRADE_TARGET_WINDOWS_RT";
    #endif
    #ifdef CORRADE_TARGET_EMSCRIPTEN
    os << "    CORRADE_TARGET_EMSCRIPTEN (" << Debug::nospace
        << __EMSCRIPTEN_major__ << Debug::nospace << "." << Debug::nospace
        << __EMSCRIPTEN_minor__ << Debug::nospace << "." << Debug::nospace
        << __EMSCRIPTEN_tiny__ << Debug::nospace << ")";
    #endif
    #ifdef CORRADE_TARGET_ANDROID
    os << "    CORRADE_TARGET_ANDROID";
    #endif
    #ifdef CORRADE_TARGET_X86
    os << "    CORRADE_TARGET_X86";
    #endif
    #ifdef CORRADE_TARGET_ARM
    os << "    CORRADE_TARGET_ARM";
    #endif
    #ifdef CORRADE_TARGET_POWERPC
    os << "    CORRADE_TARGET_POWERPC";
    #endif
    #ifdef CORRADE_TARGET_LIBCXX
    os << "    CORRADE_TARGET_LIBCXX";
    #endif
    #ifdef CORRADE_TARGET_DINKUMWARE
    os << "    CORRADE_TARGET_DINKUMWARE";
    #endif
    #ifdef CORRADE_TARGET_LIBSTDCXX
    os << "    CORRADE_TARGET_LIBSTDCXX";
    #endif
    #ifdef CORRADE_PLUGINMANAGER_NO_DYNAMIC_PLUGIN_SUPPORT
    os << "    CORRADE_PLUGINMANAGER_NO_DYNAMIC_PLUGIN_SUPPORT";
    #endif
    #ifdef CORRADE_TESTSUITE_TARGET_XCTEST
    os << "    CORRADE_TESTSUITE_TARGET_XCTEST";
    #endif
    #ifdef CORRADE_UTILITY_USE_ANSI_COLORS
    os << "    CORRADE_UTILITY_USE_ANSI_COLORS";
    #endif
    #ifdef MAGNUM_BUILD_DEPRECATED
    os << "    MAGNUM_BUILD_DEPRECATED";
    #endif
    #ifdef MAGNUM_BUILD_STATIC
    os << "    MAGNUM_BUILD_STATIC";
    #endif
    #ifdef MAGNUM_TARGET_GLES
    os << "    MAGNUM_TARGET_GLES";
    #endif
    #ifdef MAGNUM_TARGET_GLES2
    os << "    MAGNUM_TARGET_GLES2";
    #endif
    #ifdef MAGNUM_TARGET_DESKTOP_GLES
    os << "    MAGNUM_TARGET_DESKTOP_GLES";
    #endif
    #ifdef MAGNUM_TARGET_WEBGL
    os << "    MAGNUM_TARGET_WEBGL";
    #endif
    #ifdef MAGNUM_TARGET_HEADLESS
    os << "    MAGNUM_TARGET_HEADLESS";
    #endif
    os << "";

    /* Create context here, so the context creation info is displayed at proper
       place */

    GL::Context& c = GL::Context::current();

    os << "";

    #ifndef MAGNUM_TARGET_GLES
    os << "Core profile:" << (c.isCoreProfile() ? "yes" : "no");
    #endif
    #ifndef MAGNUM_TARGET_WEBGL
    //os << "Context flags:" << c.flags();
    #endif
    //os << "Detected driver:" << c.detectedDriver();

    os << "Supported GLSL versions:";
    os << "   " << Utility::String::joinWithoutEmptyParts(c.shadingLanguageVersionStrings(), ", ");

    if(args.isSet("extension-strings")) {
        os << "Extension strings:" << std::endl;
        for(auto s : c.extensionStrings()) {
            os <<  s << ", ";
        }
        return os.str();
    }

    if(args.isSet("short")) return os.str();

    os << "";

    /* Get first future (not supported) version */
    std::vector<GL::Version> versions{
        #ifndef MAGNUM_TARGET_GLES
        GL::Version::GL300,
        GL::Version::GL310,
        GL::Version::GL320,
        GL::Version::GL330,
        GL::Version::GL400,
        GL::Version::GL410,
        GL::Version::GL420,
        GL::Version::GL430,
        GL::Version::GL440,
        GL::Version::GL450,
        GL::Version::GL460,
        #else
        GL::Version::GLES300,
        #ifndef MAGNUM_TARGET_WEBGL
        GL::Version::GLES310,
        GL::Version::GLES320,
        #endif
        #endif
        GL::Version::None
    };
    std::size_t future = 0;

    if(!args.isSet("all-extensions"))
        while(versions[future] != GL::Version::None && c.isVersionSupported(versions[future]))
            ++future;

    /* Display supported OpenGL extensions from unsupported versions */
    for(std::size_t i = future; i != versions.size(); ++i) {
        if(versions[i] != GL::Version::None)
            Debug() << versions[i] << "extension support:";
        else Debug() << "Vendor extension support:";

        for(const auto& extension: GL::Extension::extensions(versions[i])) {
            std::string extensionName = extension.string();
            Debug d;
            d << "   " << extensionName << std::string(60-extensionName.size(), ' ');
            if(c.isExtensionSupported(extension))
                d << "SUPPORTED";
            else if(c.isExtensionDisabled(extension))
                d << " removed";
            else if(c.isVersionSupported(extension.requiredVersion()))
                d << "    -";
            else
                d << "   n/a";
        }

        Debug() << "";
    }

    if(!args.isSet("limits")) return os.str();

    /* Limits and implementation-defined values */
    #define _h(val) Debug() << "\n " << GL::Extensions::val::string() + std::string(":");
    #define _l(val) Debug() << "   " << #val << (sizeof(#val) > 64 ? "\n" + std::string(68, ' ') : std::string(64 - sizeof(#val), ' ')) << val;
    #define _lvec(val) Debug() << "   " << #val << (sizeof(#val) > 42 ? "\n" + std::string(46, ' ') : std::string(42 - sizeof(#val), ' ')) << val;

    Debug() << "Limits and implementation-defined values:";
    _lvec(GL::AbstractFramebuffer::maxViewportSize())
    _l(GL::AbstractFramebuffer::maxDrawBuffers())
    _l(GL::Framebuffer::maxColorAttachments())
    _l(GL::Mesh::maxVertexAttributeStride())
    #ifndef MAGNUM_TARGET_GLES2
    _l(GL::Mesh::maxElementIndex())
    _l(GL::Mesh::maxElementsIndices())
    _l(GL::Mesh::maxElementsVertices())
    #endif
    _lvec(GL::Renderer::lineWidthRange())
    _l(GL::Renderbuffer::maxSize())
    #if !(defined(MAGNUM_TARGET_WEBGL) && defined(MAGNUM_TARGET_GLES2))
    _l(GL::Renderbuffer::maxSamples())
    #endif
    _l(GL::Shader::maxVertexOutputComponents())
    _l(GL::Shader::maxFragmentInputComponents())
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Vertex))
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::TessellationControl))
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::TessellationEvaluation))
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Geometry))
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Compute))
    #endif
    _l(GL::Shader::maxTextureImageUnits(GL::Shader::Type::Fragment))
    _l(GL::Shader::maxCombinedTextureImageUnits())
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Vertex))
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::TessellationControl))
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::TessellationEvaluation))
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Geometry))
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Compute))
    #endif
    _l(GL::Shader::maxUniformComponents(GL::Shader::Type::Fragment))
    _l(GL::AbstractShaderProgram::maxVertexAttributes())
    #ifndef MAGNUM_TARGET_GLES2
    _l(GL::AbstractTexture::maxLodBias())
    #endif
    #ifndef MAGNUM_TARGET_GLES
    _lvec(GL::Texture1D::maxSize())
    #endif
    _lvec(GL::Texture2D::maxSize())
    #ifndef MAGNUM_TARGET_GLES2
    _lvec(GL::Texture3D::maxSize()) /* Checked ES2 version below */
    #endif
    _lvec(GL::CubeMapTexture::maxSize())

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::blend_func_extended>()) {
        _h(ARB::blend_func_extended)

        _l(GL::AbstractFramebuffer::maxDualSourceDrawBuffers())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::compute_shader>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::compute_shader)
        #endif

        _l(GL::AbstractShaderProgram::maxComputeSharedMemorySize())
        _l(GL::AbstractShaderProgram::maxComputeWorkGroupInvocations())
        _lvec(GL::AbstractShaderProgram::maxComputeWorkGroupCount())
        _lvec(GL::AbstractShaderProgram::maxComputeWorkGroupSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::explicit_uniform_location)
        #endif

        _l(GL::AbstractShaderProgram::maxUniformLocations())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::map_buffer_alignment>()) {
        _h(ARB::map_buffer_alignment)

        _l(GL::Buffer::minMapAlignment())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::shader_atomic_counters>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::shader_atomic_counters)
        #endif

        _l(GL::Buffer::maxAtomicCounterBindings())
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Compute))
        _l(GL::Shader::maxAtomicCounterBuffers(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedAtomicCounterBuffers())
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Compute))
        _l(GL::Shader::maxAtomicCounters(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedAtomicCounters())
        _l(GL::AbstractShaderProgram::maxAtomicCounterBufferSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::shader_image_load_store>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::shader_image_load_store)
        #endif

        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Compute))
        _l(GL::Shader::maxImageUniforms(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedImageUniforms())
        _l(GL::AbstractShaderProgram::maxCombinedShaderOutputResources())
        _l(GL::AbstractShaderProgram::maxImageUnits())
        #ifndef MAGNUM_TARGET_GLES
        _l(GL::AbstractShaderProgram::maxImageSamples())
        #endif
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::shader_storage_buffer_object>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::shader_storage_buffer_object)
        #endif

        _l(GL::Buffer::shaderStorageOffsetAlignment())
        _l(GL::Buffer::maxShaderStorageBindings())
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Vertex))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Compute))
        _l(GL::Shader::maxShaderStorageBlocks(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedShaderStorageBlocks())
        /* AbstractShaderProgram::maxCombinedShaderOutputResources() already in shader_image_load_store */
        _l(GL::AbstractShaderProgram::maxShaderStorageBlockSize())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_multisample>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_multisample)
        #endif

        _l(GL::AbstractTexture::maxColorSamples())
        _l(GL::AbstractTexture::maxDepthSamples())
        _l(GL::AbstractTexture::maxIntegerSamples())
        _lvec(GL::MultisampleTexture2D::maxSize())
        _lvec(GL::MultisampleTexture2DArray::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_rectangle>()) {
        _h(ARB::texture_rectangle)

        _lvec(GL::RectangleTexture::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES2
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::uniform_buffer_object>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::uniform_buffer_object)
        #endif

        _l(GL::Buffer::uniformOffsetAlignment())
        _l(GL::Buffer::maxUniformBindings())
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Vertex))
        #ifndef MAGNUM_TARGET_WEBGL
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Compute))
        #endif
        _l(GL::Shader::maxUniformBlocks(GL::Shader::Type::Fragment))
        _l(GL::Shader::maxCombinedUniformBlocks())
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Vertex))
        #ifndef MAGNUM_TARGET_WEBGL
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::TessellationControl))
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::TessellationEvaluation))
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Geometry))
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Compute))
        #endif
        _l(GL::Shader::maxCombinedUniformComponents(GL::Shader::Type::Fragment))
        _l(GL::AbstractShaderProgram::maxUniformBlockSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::EXT::gpu_shader4>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(EXT::gpu_shader4)
        #endif

        _l(GL::AbstractShaderProgram::minTexelOffset())
        _l(GL::AbstractShaderProgram::maxTexelOffset())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_array>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(EXT::texture_array)
        #endif

        #ifndef MAGNUM_TARGET_GLES
        _lvec(GL::Texture1DArray::maxSize())
        #endif
        _lvec(GL::Texture2DArray::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES2
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::EXT::transform_feedback>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(EXT::transform_feedback)
        #endif

        _l(GL::TransformFeedback::maxInterleavedComponents())
        _l(GL::TransformFeedback::maxSeparateAttributes())
        _l(GL::TransformFeedback::maxSeparateComponents())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::transform_feedback3>()) {
        _h(ARB::transform_feedback3)

        _l(GL::TransformFeedback::maxBuffers())
        _l(GL::TransformFeedback::maxVertexStreams())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::geometry_shader4>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::geometry_shader>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::geometry_shader4)
        #else
        _h(EXT::geometry_shader)
        #endif

        _l(GL::AbstractShaderProgram::maxGeometryOutputVertices())
        _l(GL::Shader::maxGeometryInputComponents())
        _l(GL::Shader::maxGeometryOutputComponents())
        _l(GL::Shader::maxGeometryTotalOutputComponents())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::tessellation_shader>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::tessellation_shader>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::tessellation_shader)
        #else
        _h(EXT::tessellation_shader)
        #endif

        _l(GL::Shader::maxTessellationControlInputComponents())
        _l(GL::Shader::maxTessellationControlOutputComponents())
        _l(GL::Shader::maxTessellationControlTotalOutputComponents())
        _l(GL::Shader::maxTessellationEvaluationInputComponents())
        _l(GL::Shader::maxTessellationEvaluationOutputComponents())
        _l(GL::Renderer::maxPatchVertexCount())
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_buffer_object>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_buffer>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_buffer_object)
        #else
        _h(EXT::texture_buffer)
        #endif

        _l(GL::BufferTexture::maxSize())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_buffer_range>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_buffer>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_buffer_range)
        #else
        /* Header added above */
        #endif

        _l(GL::BufferTexture::offsetAlignment())
    }

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_cube_map_array>())
    #else
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_cube_map_array>())
    #endif
    {
        #ifndef MAGNUM_TARGET_GLES
        _h(ARB::texture_cube_map_array)
        #else
        _h(EXT::texture_cube_map_array)
        #endif

        _lvec(GL::CubeMapTextureArray::maxSize())
    }
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(c.isExtensionSupported<GL::Extensions::ARB::texture_filter_anisotropic>()) {
        _h(ARB::texture_filter_anisotropic)

        _l(GL::Sampler::maxMaxAnisotropy())
    } else
    #endif
    if(c.isExtensionSupported<GL::Extensions::EXT::texture_filter_anisotropic>()) {
        _h(EXT::texture_filter_anisotropic)

        _l(GL::Sampler::maxMaxAnisotropy())
    }

    #ifndef MAGNUM_TARGET_WEBGL
    if(c.isExtensionSupported<GL::Extensions::KHR::debug>()) {
        _h(KHR::debug)

        _l(GL::AbstractObject::maxLabelLength())
        _l(GL::DebugOutput::maxLoggedMessages())
        _l(GL::DebugOutput::maxMessageLength())
        _l(GL::DebugGroup::maxStackDepth())
    }
    #endif

    #if defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    if(c.isExtensionSupported<GL::Extensions::OES::texture_3D>()) {
        _h(OES::texture_3D)

        _lvec(GL::Texture3D::maxSize())
    }
    #endif

    #undef _l
    #undef _h

    return os.str();
}

PyObject *not_initialized_error() {
    PyErr_SetString((PyObject*)&MxSimulator_Type, "simulator not initialized");
    Py_RETURN_NONE;
}
