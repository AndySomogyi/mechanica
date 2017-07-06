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
layout (location = 0) in vec2 pos;
layout (location = 1) in vec3 color;
layout (location = 2) in vec2 offset;

out vec3 fColor;

void main()
{
    gl_Position = vec4(pos * (gl_InstanceID / 100.0) + offset, 0.0, 1.0);
    fColor = color;
} 
)";

const GLchar* fragSrc = R"(
out vec4 FragColor;
  
in vec3 fColor;

void main()
{
    FragColor = vec4(fColor, 1.0);
}
)";

// Triangle vertex is standard vertex type, i.e. per vertex,
// we don't put the instanced attributes here.
struct TriangleVertex {
    Vector2 pos;
    Color3 color;
};

typedef Attribute<0, Vector2> PosAttr;
typedef Attribute<1, Color3> ColorAttr;
typedef Attribute<2, Vector2> OffsetAttr;

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


class InstancedQuads: public Platform::GlfwApplication {
    public:
        explicit InstancedQuads(const Arguments& arguments);

    private:
        void drawEvent() override;

        Buffer vertexBuffer;
        Buffer offsetBuffer;
        Mesh mesh;
        ShaderProgram shaderProgram;
};

InstancedQuads::InstancedQuads(const Arguments& arguments) :
        Platform::GlfwApplication{arguments, Configuration{}.
            setVersion(Version::GL410).
            setTitle("Instanced Drawing Example")} {

    static const TriangleVertex vertices[] = {
        // positions     // colors
        {{-0.05f,  0.05f},  {1.0f, 0.0f, 0.0f}},
        {{ 0.05f, -0.05f},  {0.0f, 1.0f, 0.0f}},
        {{-0.05f, -0.05f},  {0.0f, 0.0f, 1.0f}},

        {{-0.05f,  0.05f},  {1.0f, 0.0f, 0.0f}},
        {{ 0.05f, -0.05f},  {0.0f, 1.0f, 0.0f}},
        {{ 0.05f,  0.05f},  {0.0f, 1.0f, 1.0f}}
    };

   vertexBuffer.setData(vertices, BufferUsage::StaticDraw);

   Vector2 translations[100];

    int index = 0;
    float offset = 0.1f;
    for (int y = -10; y < 10; y += 2)
    {
        for (int x = -10; x < 10; x += 2)
        {
            Vector2 translation;
            translation[0] = float(x) / 10.0f + offset;
            translation[1] = float(y) / 10.0f + offset;
            translations[index++] = translation;
        }
    }

   offsetBuffer.setData(translations, BufferUsage::StaticDraw);

   mesh.setPrimitive(MeshPrimitive::Triangles)
       .setCount(6)
       .setInstanceCount(100);

   mesh.addVertexBuffer(vertexBuffer, 0, PosAttr{}, ColorAttr{});

   // the '1' divisor tells OpenGL this is an instanced vertex attribute.
   mesh.addVertexBufferInstanced(offsetBuffer, 1 /* divisor */, 0 /* offset */, OffsetAttr{});
}

void InstancedQuads::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    mesh.draw(shaderProgram);

    swapBuffers();
}

int main(int argc, char** argv) {
    InstancedQuads app({argc, argv});
    return app.exec();
}
