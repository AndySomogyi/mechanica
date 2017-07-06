#include <Magnum/Magnum.h>
#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shader.h>
#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Context.h>
#include <Magnum/Version.h>
#include <Magnum/AbstractShaderProgram.h>


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

typedef Attribute<0, Vector2> PositionAttr;
typedef Attribute<1, Color3> ColorAttr;

class ShaderProgram : public AbstractShaderProgram {
public:

    explicit ShaderProgram() {
        MAGNUM_ASSERT_VERSION_SUPPORTED(Version::GL330);

        Shader vert{Version::GL330, Shader::Type::Vertex};
        Shader frag{Version::GL330, Shader::Type::Fragment};

        vert.addSource(vertSrc);
        frag.addSource(fragSrc);

        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, frag}));

        attachShaders({vert, frag});

        CORRADE_INTERNAL_ASSERT_OUTPUT(link());
    };
};


class Polygons: public Platform::GlfwApplication {
    public:
        explicit Polygons(const Arguments& arguments);

    private:
        void drawEvent() override;

        Buffer vertexBuffer;
        Buffer indexBuffer;
        Mesh mesh;
        ShaderProgram shaderProgram;
};

Polygons::Polygons(const Arguments& arguments) :
        Platform::GlfwApplication{arguments, Configuration{}.
            setVersion(Version::GL410).
            setTitle("Polygon Example")} {

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

   vertexBuffer.setData(vertices, BufferUsage::StaticDraw);

   indexBuffer.setData(elements, BufferUsage::StaticDraw);

   mesh.setPrimitive(MeshPrimitive::Triangles)
       .setCount(6)
       .addVertexBuffer(vertexBuffer, 0,
           PositionAttr{},
           ColorAttr{});

   mesh.setIndexBuffer(indexBuffer, 0, Mesh::IndexType::UnsignedInt);
}

void Polygons::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    mesh.draw(shaderProgram);

    swapBuffers();
}

int main(int argc, char** argv) {
    Polygons app({argc, argv});
    return app.exec();
}
