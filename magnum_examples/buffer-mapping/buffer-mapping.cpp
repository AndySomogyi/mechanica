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

#include <random>


using namespace Magnum;
using namespace Magnum::Shaders;

// Shader sources


const GLchar* fragSrc = R"(
    in vec3 Color;
    out vec4 outColor;
    void main()
    {
        outColor = vec4(Color, 1.0);
    }
)";

// the vertex shader simply passes through data
const GLchar* vertSrc = R"(
    #version 330
    layout(location = 0) in vec4 vposition;
    void main() {
       gl_Position = vposition;
    }
)";

// the geometry shader creates the billboard quads
const GLchar *geomSrc = R"("
    #version 330
    uniform mat4 View;
    uniform mat4 Projection;
    layout (points) in;
    layout (triangle_strip, max_vertices = 4) out;
    out vec2 txcoord;
    void main() {
        vec4 pos = View*gl_in[0].gl_Position;
        txcoord = vec2(-1,-1);
        gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
        EmitVertex();
        txcoord = vec2( 1,-1);
        gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
        EmitVertex();
        txcoord = vec2(-1, 1);
        gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
        EmitVertex();
        txcoord = vec2( 1, 1);
        gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
        EmitVertex();
    }
)";

// the fragment shader creates a bell like radial color distribution
const GLchar* fragSrc = R"(
    #version 330
    in vec2 txcoord;
    layout(location = 0) out vec4 FragColor;
    void main() {
        float s = 0.2*(1/(1+15.*dot(txcoord, txcoord))-1/16.);
        FragColor = s*vec4(0.3,0.3,1.0,1);
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
        Shader geom{Version::GL330, Shader::Type::Geometry};

        vert.addSource(vertSrc);
        geom.addSource(geomSrc);
        frag.addSource(fragSrc);


        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, geom, frag}));

        attachShaders({vert, geom, frag});

        CORRADE_INTERNAL_ASSERT_OUTPUT(link());

        viewLoc = uniformLocation("View");
        projLoc = uniformLocation("Projection");
    };

    void setViewMatrix(const Matrix4& mat) {
        setUniform(viewLoc, mat);
    }

    void setProjMatrix(const Matrix4& mat) {
        setUniform(projLoc, mat);
    }

private:
    int viewLoc, projLoc;
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


        static const int particles = 128*1024;

        // randomly place particles in a cube
        std::vector<Vector3> vertexData;
        std::vector<Vector3> velocity;
};

Polygons::Polygons(const Arguments& arguments) :
        Platform::GlfwApplication{arguments, Configuration{}.
            setVersion(Version::GL410).
            setTitle("Polygon Example")} {

    // Seed with a real random value, if available
    std::random_device r;

    // Choose a random mean between 1 and 6
    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> uniform_dist(-5, 5);





    for(int i = 0;i<particles;++i) {
        vertexData[i] = {uniform_dist(), 20+uniform_dist(), uniform_dist()};
    }







   //vertexBuffer.setData(vertexData. BufferUsage::DynamicDraw);

   //indexBuffer.setData(elements, BufferUsage::StaticDraw);

   //mesh.setPrimitive(MeshPrimitive::Triangles)
   //    .setCount(6)
   //    .addVertexBuffer(vertexBuffer, 0,
   //        PositionAttr{},
   //        ColorAttr{});

   //mesh.setIndexBuffer(indexBuffer, 0, Mesh::IndexType::UnsignedInt);
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
