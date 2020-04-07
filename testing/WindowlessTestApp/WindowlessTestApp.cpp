/*
 * WindowlessTestApp.cpp
 *
 *  Created on: Mar 22, 2019
 *      Author: andy
 */

#include <MxWindowless.h>

#include <Magnum/DebugTools/Screenshot.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/Shaders/VertexColor.h>
#include <MagnumPlugins/TgaImageConverter/TgaImageConverter.h>
#include <Magnum/Math/Color.h>

#include <Magnum/PixelFormat.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>


using namespace Magnum;
using namespace Magnum::Trade;

class MyApplication: public Platform::WindowlessApplication {
    public:
        using Platform::WindowlessApplication::WindowlessApplication;

        int exec() override;
};

int MyApplication::exec() {
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
    shader.draw(mesh);

    const GL::PixelFormat format = framebuffer.implementationColorReadFormat();
    Image2D image = framebuffer.read(framebuffer.viewport(), PixelFormat::RGBA8Unorm);
    
    TgaImageConverter conv;

    MutableImageView2D im(image);

    conv.exportToFile(im, "triangle.tga");

    return 0;

}

MAGNUM_WINDOWLESSAPPLICATION_MAIN(MyApplication)




