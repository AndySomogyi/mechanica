/*
 * MxApplication.cpp
 *
 *  Created on: Mar 27, 2019
 *      Author: andy
 */

#include <rendering/MxApplication.h>
#include <rendering/MxWindowlessApplication.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>

#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Magnum/Animation/Easing.h>


#include <Corrade/Utility/Directory.h>
#include <rendering/MxImageConverters.h>
#include <Magnum/Math/Color.h>

using namespace Magnum;

#include <iostream>


static void _test_draw(Magnum::GL::AbstractFramebuffer &framebuffer) {
    
    framebuffer
    .clear(GL::FramebufferClear::Color)
    .bind();
    
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
    
  
    
    Shaders::VertexColor2D shader;
    shader.draw(mesh);
}

PyObject* MxTestImage(PyObject *module, PyObject* self, PyObject* args) {
    
    GL::Renderbuffer renderbuffer;
    renderbuffer.setStorage(GL::RenderbufferFormat::RGBA8, {640, 480});
    GL::Framebuffer framebuffer{{{}, {640, 480}}};
    
    framebuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, renderbuffer);
 
    _test_draw(framebuffer);

    const GL::PixelFormat format = framebuffer.implementationColorReadFormat();
    Image2D image = framebuffer.read(framebuffer.viewport(), PixelFormat::RGBA8Unorm);


    auto jpegData = convertImageDataToJpeg(image);


    /* Open file */
    if(!Utility::Directory::write("triangle.jpg", jpegData)) {
        Error() << "Trade::AbstractImageConverter::exportToFile(): cannot write to file" << "triangle.jpg";
        return NULL;
    }

    return PyBytes_FromStringAndSize(jpegData.data(), jpegData.size());
}




PyObject* MxFramebufferImageData(PyObject *module, PyObject* self, PyObject* args) {
    
    MxSimulator *sim = MxSimulator_Get();
    
    Magnum::GL::AbstractFramebuffer &framebuffer = sim->app->framebuffer();
    
    sim->app->redraw();
    
    //_test_draw(framebuffer);
    
    Image2D image = framebuffer.read(framebuffer.viewport(), PixelFormat::RGBA8Unorm);
    
    auto jpegData = convertImageDataToJpeg(image);
    
    return PyBytes_FromStringAndSize(jpegData.data(), jpegData.size());
}

HRESULT MxApplication::simulationStep() {
    
    /* Pause simulation if the mouse was pressed (camera is moving around).
     This avoid freezing GUI while running the simulation */
    
    /*
     if(!_pausedSimulation && !_mousePressed) {
     // Adjust the substep number to maximize CPU usage each frame
     const Float lastAvgStepTime = _timeline.previousFrameDuration()/Float(_substeps);
     const Int newSubsteps = lastAvgStepTime > 0 ? Int(1.0f/60.0f/lastAvgStepTime) + 1 : 1;
     if(Math::abs(newSubsteps - _substeps) > 1) _substeps = newSubsteps;
     
     // TODO: move substeps to universe step.
     if(MxUniverse_Flag(MxUniverse_Flags::MX_RUNNING)) {
     for(Int i = 0; i < _substeps; ++i) {
     MxUniverse_Step(0, 0);
     }
     }
     }
     */
    
    static Float offset = 0.0f;
    if(_dynamicBoundary) {
        /* Change fluid boundary */
        static Float step = 2.0e-3f;
        if(_boundaryOffset > 1.0f || _boundaryOffset < 0.0f) {
            step *= -1.0f;
        }
        _boundaryOffset += step;
        offset = Math::lerp(0.0f, 0.5f, Animation::Easing::quadraticInOut(_boundaryOffset));
    }
    
    currentStep += 1;
    
    // TODO: get rid of this
    return  MxUniverse_Step(0,0);
}

HRESULT MxApplication::run()
{
    std::cout << MX_FUNCTION << std::endl;
    MxUniverse_SetFlag(MX_RUNNING, true);
    return messageLoop();
}




