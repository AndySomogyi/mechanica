#include <Magnum/Magnum.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Math/Color.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/AbstractShaderProgram.h>

#include <Magnum/Magnum.h>
#include <Corrade/Utility/Resource.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Math/Color.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/AbstractShaderProgram.h>

#include <Corrade/Containers/Reference.h>



using namespace Magnum;
using namespace Magnum::Shaders;

// Shader sources
const GLchar* vertSrc = R"(
    layout(location=0) in vec2 position;
    layout(location=1) in vec3 color;
    out vec3 Color;
    void main()
    {
        Color = color;
        gl_Position = vec4(position, 0.0, 1.0);
    }
)";

const GLchar* fragSrc = R"(
    in vec3 Color;
    out vec4 outColor;
    void main()
    {
        outColor = vec4(Color, 1.0);
    }
)";

struct TriangleVertex {
    Vector2 position;
    Color3 color;
};

typedef GL::Attribute<0, Vector2> PositionAttr;
typedef GL::Attribute<1, Color3> ColorAttr;

class ShaderProgram : public GL::AbstractShaderProgram {
public:

    explicit ShaderProgram() {
        MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL330);

        GL::Shader vert{GL::Version::GL330, GL::Shader::Type::Vertex};
        GL::Shader frag{GL::Version::GL330, GL::Shader::Type::Fragment};

        vert.addSource(vertSrc);
        frag.addSource(fragSrc);

        CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

        attachShaders({vert, frag});

        CORRADE_INTERNAL_ASSERT_OUTPUT(link());
    };
};


class Polygons: public Platform::GlfwApplication {
    public:
        explicit Polygons(const Arguments& arguments);

    private:
        void drawEvent() override;

        GL::Buffer vertexBuffer;
        GL::Buffer indexBuffer;
        GL::Mesh mesh;
        ShaderProgram shaderProgram;
};

Polygons::Polygons(const Arguments& arguments) :
        Platform::GlfwApplication{arguments, Configuration{}.setTitle("Polygon Example"),
    GLConfiguration{}.setVersion(GL::Version::GL410)} {

   static const TriangleVertex vertices[] = {
   //  position         color
       {{-0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}}, // Top-left
       {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}}, // Top-right
       {{ 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}}, // Bottom-right
       {{-0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}}  // Bottom-left
   };

   static const GLuint elements[] = {
       0, 1, 2,
       2, 3, 0
   };

   vertexBuffer.setData(vertices, GL::BufferUsage::StaticDraw);

   indexBuffer.setData(elements, GL::BufferUsage::StaticDraw);

   mesh.setPrimitive(MeshPrimitive::Triangles)
       .setCount(6)
       .addVertexBuffer(vertexBuffer, 0,
           PositionAttr{},
           ColorAttr{});

   mesh.setIndexBuffer(indexBuffer, 0, GL::Mesh::IndexType::UnsignedInt);
}

void Polygons::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);

    mesh.draw(shaderProgram);

    swapBuffers();
}

int main(int argc, char** argv) {
    Polygons app({argc, argv});
    return app.exec();
}
