/*
 * WindowlessTestApp.cpp
 *
 *  Created on: Mar 22, 2019
 *      Author: andy
 */



#include <Magnum/DebugTools/Screenshot.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/Shaders/VertexColor.h>


#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>

#include <rendering/MxApplication.h>
#include <MxSurfaceSimulator.h>
#include <Corrade/Utility/Directory.h>
#include <rendering/MxImageConverters.h>

using namespace Magnum;
using namespace Magnum::Trade;

#include <iostream>



static int exec() {
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

    Debug{} << "test framebuffer size: " << framebuffer.viewport();

    Shaders::VertexColor2D shader;
    shader.draw(mesh);

    const GL::PixelFormat format = framebuffer.implementationColorReadFormat();
    Image2D image = framebuffer.read(framebuffer.viewport(), PixelFormat::RGBA8Unorm);
    
    auto jpegData = convertImageDataToJpeg(image);

    /* Open file */
    if(!Utility::Directory::write("triangle.jpg", jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
    }

    return 0;
}


#define OFFSET(f) std::cout << "offset: " << offsetof(CType, f) << ", " << offsetof(PyTypeObject, f) << std::endl;

int main (int argc, char** argv) {

    std::cout << "CType: " << sizeof(CType) << ", PyObjectType: " << sizeof(PyTypeObject) << std::endl;

    OFFSET(tp_name);

    Mx_Initialize(0);

    const std::string dirName = MX_MODEL_DIR;

    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t2.obj";
    //const char* fileName = "cylinder.1.obj";
    //const char* fileName = "cube1.obj";
    const char* fileName = "hex_cylinder.1.obj";
    //const char* fileName = "football.t1.obj";
    //const char* fileName = "football.t1.obj";


    std::string modelPath = dirName  + "/" + fileName;

    MxApplicationConfig conf = {};

    PyObject *app = MxApplication_New(argc, argv, &conf);


    MxSurfaceSimulator_Config simConf = {{600,900}, modelPath.c_str()};
    MxSurfaceSimulator *sim = MxSurfaceSimulator_New(&simConf);

    MxSurfaceSimulator_ImageData(sim, "surfacesimulator.jpg");


    //MxApplication::create(argc, argv, {});

    exec();

    Py_DECREF(app);

    MxApplication::destroy();

}






