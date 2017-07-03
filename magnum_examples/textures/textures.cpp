/*
    This file is part of Magnum.

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017 —
            Vladimír Vondruš <mosra@centrum.cz>

    This is free and unencumbered software released into the public domain.

    Anyone is free to copy, modify, publish, use, compile, sell, or distribute
    this software, either in source code form or as a compiled binary, for any
    purpose, commercial or non-commercial, and by any means.

    In jurisdictions that recognize copyright laws, the author or authors of
    this software dedicate any and all copyright interest in the software to
    the public domain. We make this dedication for the benefit of the public
    at large and to the detriment of our heirs and successors. We intend this
    dedication to be an overt act of relinquishment in perpetuity of all
    present and future rights to this software under copyright law.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shaders/VertexColor.h>
#include <MagnumPlugins/PngImporter/PngImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Texture.h>
#include <Magnum/TextureFormat.h>


using namespace Magnum;
using namespace Magnum::Trade;

class TriangleExample: public Platform::Application {
    public:
        explicit TriangleExample(const Arguments& arguments);

    private:
        void drawEvent() override;

        Buffer _buffer;
        Mesh _mesh;
        Shaders::VertexColor2D _shader;

        Texture2D kittenTexture;

        Texture2D puppyTexture;
};

TriangleExample::TriangleExample(const Arguments& arguments): Platform::Application{arguments, Configuration{}.setTitle("Magnum Triangle Example")} {
    using namespace Math::Literals;






    PngImporter importer;

    importer.openFile("kitten.png");

    auto kittenImg = importer.image2D(0);


   importer.openFile("puppy.png");
   auto puppyImg = importer.image2D(0);


   kittenTexture.setWrapping(Sampler::Wrapping::ClampToEdge)
           .setMagnificationFilter(Sampler::Filter::Linear)
           .setMinificationFilter(Sampler::Filter::Linear)
           .setStorage(1, TextureFormat::RGB8, kittenImg->size())
           .setSubImage(0, {}, *kittenImg);

   puppyTexture.setWrapping(Sampler::Wrapping::ClampToEdge)
              .setMagnificationFilter(Sampler::Filter::Linear)
              .setMinificationFilter(Sampler::Filter::Linear)
              .setStorage(1, TextureFormat::RGB8, puppyImg->size())
              .setSubImage(0, {}, *puppyImg);

   struct TriangleVertex {
       Vector2 position;
       Color3 color;
       Vector2 textureCoordinates;
   };


   static const TriangleVertex data[]{
   //  Position      Color             Texcoords
       {{-0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}}, // Top-left
       {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}}, // Top-right
       {{ 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}, // Bottom-right
       {{-0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}  // Bottom-left
   };

   _buffer.setData(data, BufferUsage::StaticDraw);
   _mesh.setPrimitive(MeshPrimitive::Triangles)
       .setCount(4)
       .addVertexBuffer(_buffer, 0,
           Shaders::VertexColor2D::Position{},
           Shaders::VertexColor2D::Color{Shaders::VertexColor2D::Color::Components::Three});


}

void TriangleExample::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    _mesh.draw(_shader);

    swapBuffers();
}


MAGNUM_APPLICATION_MAIN(TriangleExample)
