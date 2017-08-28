#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <MagnumPlugins/PngImporter/PngImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Texture.h>
#include <Magnum/TextureFormat.h>
#include <Magnum/Context.h>
#include <Magnum/Shader.h>
#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Version.h>
#include <Magnum/AbstractShaderProgram.h>
#include <chrono>

#include <iostream>


using namespace Magnum;
using namespace Magnum::Trade;
using namespace Magnum::Shaders;

typedef Attribute<0, Vector2> PositionAttr;
typedef Attribute<1, Color3> ColorAttr;
typedef Attribute<2, Vector2> TextureAttr;

struct TriangleVertex {
    Vector2 position;
    Color3 color;
    Vector2 texcoord;
};

// Shader sources
const GLchar* vertSrc = R"(
layout(location=0) in vec2 position;
layout(location=1) in vec3 color;
layout(location=2) in vec2 texcoord;
out vec3 Color;
out vec2 Texcoord;

uniform mat4 proj;
uniform mat4 model;
uniform mat4 view;

void main()
{
    Color = color;
    Texcoord = texcoord;

    gl_Position = view * model * proj * vec4(position, 0.0, 1.0);

    }
)";

const GLchar* fragSrc = R"(
in vec3 Color;
in vec2 Texcoord;
out vec4 outColor;
uniform sampler2D texKitten;
uniform sampler2D texPuppy;

void main()
{
    outColor = mix(texture(texKitten, Texcoord), texture(texPuppy, Texcoord), 0.5);
}
)";

class TexturedShader: public AbstractShaderProgram {
public:

    Texture2D kittenTexture;

    Texture2D puppyTexture;

    explicit TexturedShader() {

        // load the vertex and fragment shaders
        MAGNUM_ASSERT_VERSION_SUPPORTED(Version::GL330);

        Shader vert{Version::GL330, Shader::Type::Vertex};
        Shader frag{Version::GL330, Shader::Type::Fragment};

        vert.addSource(vertSrc);
        frag.addSource(fragSrc);

        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, frag}));

        attachShaders({vert,  frag});

        CORRADE_INTERNAL_ASSERT_OUTPUT(link());

        /* Load the texture */
        const Utility::Resource rs{"data"};

        PngImporter importer;

        if(!importer.openData(rs.getRaw("kitten.png"))) {
            std::cout << "could not open kitten.png resource";
            exit(2);
        }
        auto kittenImg = importer.image2D(0);

        if(!importer.openData(rs.getRaw("puppy.png"))) {
            std::cout << "could not open puppy.png resource";
            exit(2);
        }

        auto puppyImg = importer.image2D(0);

        kittenTexture.setWrapping(Sampler::Wrapping::ClampToEdge)
        .setMagnificationFilter(Sampler::Filter::Linear)
        .setMinificationFilter(Sampler::Filter::Linear)
        .setStorage(1, TextureFormat::RGB8, kittenImg->size())
        .setSubImage(0, {}, *kittenImg)
        .bind(0);
        setUniform(uniformLocation("texKitten"), 0);

        puppyTexture.setWrapping(Sampler::Wrapping::ClampToEdge)
        .setMagnificationFilter(Sampler::Filter::Linear)
        .setMinificationFilter(Sampler::Filter::Linear)
        .setStorage(1, TextureFormat::RGB8, puppyImg->size())
        .setSubImage(0, {}, *puppyImg)
        .bind(1);
        setUniform(uniformLocation("texPuppy"), 1);


        projLoc = uniformLocation("proj");
        modelLoc = uniformLocation("model");
        viewLoc = uniformLocation("view");
    }

    void setProj(const Matrix4& mat) {
        setUniform(projLoc, mat);
    }

    void setModel(const Matrix4& mat) {
        setUniform(modelLoc, mat);
    }

    void setView(const Matrix4& mat) {
        setUniform(viewLoc, mat);
    }

private:
    int projLoc;
    int modelLoc;
    int viewLoc;
};

class transformation3D: public Platform::GlfwApplication {
public:
    explicit transformation3D(const Arguments& arguments);

private:
    void drawEvent() override;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Mesh mesh;
    TexturedShader shader;

    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

transformation3D::transformation3D(const Arguments& arguments) :
Platform::GlfwApplication{
    arguments,
    Configuration{}
    .setVersion(Version::GL410)
    .setTitle("Polygon Example")} {

        static const TriangleVertex vertices[]{
        //      Position            Color         Texcoords
            {{-0.5f,  0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}}, // Top-left
            {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}}, // Top-right
            {{ 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}}, // Bottom-right
            {{-0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f}}  // Bottom-left
        };

        static const GLuint elements[] = {
            0, 1, 2,
            2, 3, 0
        };

        Matrix4 view = Matrix4::lookAt(Vector3{1.2f, 1.2f, 1.2f},
                                       Vector3{0.0f, 0.0f, 0.0f},
                                       Vector3{0.0f, 0.0f, 1.0f});
        shader.setView(view);

        Matrix4 proj = Matrix4::rotationZ(Rad{3.141592f});
        shader.setProj(proj);

        Matrix4 model = Math::IdentityInit;
        shader.setModel(model);

        vertexBuffer.setData(vertices, BufferUsage::StaticDraw);

        indexBuffer.setData(elements, BufferUsage::StaticDraw);

        mesh.setPrimitive(MeshPrimitive::Triangles)
        .setCount(6)
        .addVertexBuffer(vertexBuffer, 0,
                         PositionAttr{},
                         ColorAttr{},
                         TextureAttr{});

        mesh.setIndexBuffer(indexBuffer, 0, Mesh::IndexType::UnsignedInt);


        start = std::chrono::high_resolution_clock::now();
    }

void transformation3D::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    // Calculate transformation
    auto now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::duration<float>>(now - start).count();


    auto model = Matrix4::rotation(Rad{time * Constants::pi()}, Vector3{0.0f, 0.0f, 1.0f});

    shader.setModel(model);


    mesh.draw(shader);

    redraw();

    swapBuffers();
}


MAGNUM_APPLICATION_MAIN(transformation3D)

