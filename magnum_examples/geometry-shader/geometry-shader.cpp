#include <Magnum/Magnum.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Math/Color.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/AbstractShaderProgram.h>
#include <chrono>

using namespace Magnum;
using namespace Magnum::Shaders;

struct TriangleVertex {
    Vector3 pos;
    Color3 color;
};

typedef Attribute<0, Vector3> PosAttr;
typedef Attribute<1, Color3> ColorAttr;

// Vertex shader
const GLchar* vertSrc = R"(
    layout(location=0) in vec3 pos;
    layout(location=1) in vec3 color;

    out vec3 vColor;

    void main()
    {
        gl_Position = vec4(pos, 1.0);
        vColor = color;
    }
)";

// Geometry shader
const GLchar* geomSrc = R"(
    layout(points) in;
    layout(line_strip, max_vertices = 16) out;

    in vec3 vColor[];
    out vec3 fColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    void main()
    {
        fColor = vColor[0];

        gl_Position = proj * view * model * gl_in[0].gl_Position;

        // +X direction is "North", -X direction is "South"
        // +Y direction is "Up",    -Y direction is "Down"
        // +Z direction is "East",  -Z direction is "West"
        //                                     N/S   U/D   E/W
        vec4 NEU = proj * view * model * vec4( 0.1,  0.1,  0.1, 0.0);
        vec4 NED = proj * view * model * vec4( 0.1, -0.1,  0.1, 0.0);
        vec4 NWU = proj * view * model * vec4( 0.1,  0.1, -0.1, 0.0);
        vec4 NWD = proj * view * model * vec4( 0.1, -0.1, -0.1, 0.0);
        vec4 SEU = proj * view * model * vec4(-0.1,  0.1,  0.1, 0.0);
        vec4 SED = proj * view * model * vec4(-0.1, -0.1,  0.1, 0.0);
        vec4 SWU = proj * view * model * vec4(-0.1,  0.1, -0.1, 0.0);
        vec4 SWD = proj * view * model * vec4(-0.1, -0.1, -0.1, 0.0);

        // Create a cube centered on the given point.
        gl_Position = gl_in[0].gl_Position + NED;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NWD;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SWD;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SED;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SEU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SWU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NWU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NEU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NED;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SED;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SEU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NEU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NWU;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + NWD;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SWD;
        EmitVertex();

        gl_Position = gl_in[0].gl_Position + SWU;
        EmitVertex();

        EndPrimitive();
    }
)";

// Fragment shader
const GLchar* fragSrc = R"(
    in vec3 fColor;
    out vec4 outColor;

    void main()
    {
        outColor = vec4(fColor, 1.0);
    }
)";


class ShaderProgram : public AbstractShaderProgram {
public:

    explicit ShaderProgram() {
        MAGNUM_ASSERT_VERSION_SUPPORTED(Version::GL330);

        Shader vert{Version::GL330, Shader::Type::Vertex};
        Shader geom{Version::GL330, Shader::Type::Geometry};
        Shader frag{Version::GL330, Shader::Type::Fragment};

        vert.addSource(vertSrc);
        geom.addSource(geomSrc);
        frag.addSource(fragSrc);

        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, geom, frag}));

        attachShaders({vert, geom, frag});

        CORRADE_INTERNAL_ASSERT_OUTPUT(link());

        modelLoc = uniformLocation("model");
        viewLoc = uniformLocation("view");
        projLoc = uniformLocation("proj");
    };


    void setModelMatrix(const Matrix4& mat) {
        setUniform(modelLoc, mat);
    }

    void setViewMatrix(const Matrix4& mat) {
        setUniform(viewLoc, mat);
    }

    void setProjMatrix(const Matrix4& mat) {
        setUniform(projLoc, mat);
    }

private:
    int modelLoc, viewLoc, projLoc;
};

class GeometryShaderExample: public Platform::GlfwApplication {
    public:
        explicit GeometryShaderExample(const Arguments& arguments);

    private:
        void drawEvent() override;

        Buffer vertexBuffer;
        Mesh mesh;
        ShaderProgram shaderProgram;
        std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
};

GeometryShaderExample::GeometryShaderExample(const Arguments& arguments) :
        Platform::GlfwApplication{arguments, Configuration{}.
            setVersion(Version::GL410).
            setTitle("Polygon Example")} {

    static const TriangleVertex vertices[] = {
        //  Coordinates             Color
        {{-0.45f,  0.45f, -0.45f}, {1.0f, 0.0f, 0.0f}},
        {{ 0.45f,  0.45f, -0.45f}, {0.0f, 1.0f, 0.0f}},
        {{ 0.45f, -0.45f, -0.45f}, {0.0f, 0.0f, 1.0f}},
        {{-0.45f, -0.45f, -0.45f}, {1.0f, 1.0f, 0.0f}},
        {{-0.45f,  0.45f,  0.45f}, {0.0f, 1.0f, 1.0f}},
        {{ 0.45f,  0.45f,  0.45f}, {1.0f, 0.0f, 1.0f}},
        {{ 0.45f, -0.45f,  0.45f}, {1.0f, 0.5f, 0.5f}},
        {{-0.45f, -0.45f,  0.45f}, {0.5f, 1.0f, 0.5f,}}
    };

   vertexBuffer.setData(vertices, BufferUsage::StaticDraw);
   mesh.setPrimitive(MeshPrimitive::Points)
       .setCount(8)
       .addVertexBuffer(vertexBuffer, 0,
           PosAttr{},
           ColorAttr{});

   Matrix4 view = Matrix4::lookAt(
       Vector3{1.5f, 1.5f, 2.0f},
       Vector3{0.0f, 0.0f, 0.0f},
       Vector3{0.0f, 0.0f, 1.0f});
   shaderProgram.setViewMatrix(view);

   Matrix4 proj = Matrix4::perspectiveProjection(Rad{45.0f}, 800.0f / 600.0f, 1.0f, 10.0f);
   shaderProgram.setProjMatrix(proj);

   Matrix4 model = Math::IdentityInit;
   shaderProgram.setModelMatrix(model);

   t_start = std::chrono::high_resolution_clock::now();
}

void GeometryShaderExample::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    // Calculate transformation
    auto t_now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

    auto model = Matrix4::rotation(
            Rad{0.25f * time * Constants::pi()},
            Vector3{0.0f, 0.0f, 1.0f}
    );

    shaderProgram.setModelMatrix(model);

    mesh.draw(shaderProgram);

    redraw();

    swapBuffers();
}

int main(int argc, char** argv) {
    GeometryShaderExample app({argc, argv});
    return app.exec();
}
