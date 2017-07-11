/* OpenGL example code - buffer mapping
 *
 * This example uses the geometry shader again for particle drawing.
 * The particles are animated on the cpu and uploaded every frame by
 * writing into a mapped vertex buffer object.  
 *
 * Original Autor: Jakob Progsch
 * Ported to Magnum: Andy Somogyi
 */

#include <Magnum/Magnum.h>
#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shader.h>
#include <Magnum/Shaders/Shaders.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Context.h>
#include <Magnum/Version.h>
#include <Magnum/AbstractShaderProgram.h>
#include <Magnum/Renderer.h>

#include <random>
#include <iostream>

using namespace Magnum;
using namespace Magnum::Shaders;


// Shader sources
// the vertex shader simply passes through data
const GLchar* vertSrc = R"(
    layout(location = 0) in vec3 vposition;
    void main() {
       gl_Position = vec4(vposition, 1.0);
    }
)";

// the geometry shader creates the billboard quads
const GLchar *geomSrc = R"(
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
    in vec2 txcoord;
    layout(location = 0) out vec4 FragColor;
    void main() {
        float s = 0.2*(1/(1+15.*dot(txcoord, txcoord))-1/16.);
        FragColor = s*vec4(0.3,0.3,1.0,1);
    }
)";



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


class BufferMapping: public Platform::GlfwApplication {
public:
    explicit BufferMapping(const Arguments& arguments);
    
private:
    void drawEvent() override;
    
    Buffer vertexBuffer;
    Mesh mesh;
    ShaderProgram shaderProgram;
    
    
    static const int particles = 128*1024;
    
    // randomly place particles in a cube
    std::vector<Vector3> vertexData;
    std::vector<Vector3> velocity;
};

BufferMapping::BufferMapping(const Arguments& arguments) :
        Platform::GlfwApplication{arguments, Configuration{}.
            setVersion(Version::GL410).
            setTitle("Buffer Mapping Example")},
            vertexData{particles},
            velocity{particles} {

    // we are blending so no depth testing
    Renderer::disable(Renderer::Feature::DepthTest);

    // enable blending
    Renderer::enable(Renderer::Feature::Blending);

    //  and set the blend function to result = 1*source + 1*destination
    Renderer::setBlendFunction(Renderer::BlendFunction::One, Renderer::BlendFunction::One);

    for(int i = 0;i<particles;++i) {
        vertexData[i] = Vector3{0.5f-float(std::rand())/RAND_MAX,
                                0.5f-float(std::rand())/RAND_MAX,
                                0.5f-float(std::rand())/RAND_MAX};
        vertexData[i] = Vector3{0.0f,20.0f,0.0f} + 5.0f*vertexData[i];
    }

    // set the vertex buffer size / initial data. Note, have to use Containers::arrayView
    // function to make an ArrayView obj.
   vertexBuffer.setData(Containers::arrayView(vertexData.data(), vertexData.size()), BufferUsage::DynamicDraw);

   mesh.setPrimitive(MeshPrimitive::Points)
       .setCount(vertexData.size())
       .addVertexBuffer(vertexBuffer, 0, Attribute<0,Vector3>{});
}

void BufferMapping::drawEvent() {

    // get the time in seconds
    float t = glfwGetTime();

    // update physics
    // define spheres for the particles to bounce off
    static const int spheres = 3;
    static const Vector3 center[] = {{0,12,1}, {-3,0,0}, {5,-10,0}};
    static const float radius[] = {3, 7, 12};


    // physical parameters
    static const float dt = 1.0f/60.0f;
    static const Vector3 g = {0.0f, -9.81f, 0.0f};
    static const float bounce = 1.2f; // inelastic: 1.0f, elastic: 2.0f


    for(int i = 0;i<particles;++i) {
        // resolve sphere collisions
        for(int j = 0;j<spheres;++j) {
            Vector3 diff = vertexData[i]-center[j];
            float dist = diff.length();
            if(dist<radius[j] && Math::dot(diff, velocity[i])<0.0f)
                velocity[i] -= bounce*diff/(dist*dist)*Math::dot(diff, velocity[i]);
        }
        // euler iteration
        velocity[i] += dt*g;
        vertexData[i] += dt*velocity[i];
        // reset particles that fall out to a starting position
        if(vertexData[i][1]<-30.0) {
            vertexData[i] = Vector3{
                0.5f-float(std::rand())/RAND_MAX,
                0.5f-float(std::rand())/RAND_MAX,
                0.5f-float(std::rand())/RAND_MAX
            };
            vertexData[i] = Vector3{0.0f,20.0f,0.0f} + 5.0f*vertexData[i];
            velocity[i] = Vector3{0,0,0};
        }
    }

    // map the buffer
    Vector3* mapped = vertexBuffer.map<Vector3>(0,  particles * sizeof(Vector3),
        Buffer::MapFlag::Write|Buffer::MapFlag::InvalidateBuffer);

    // copy data into the mapped memory
    std::copy(vertexData.begin(), vertexData.end(), mapped);

    vertexBuffer.unmap();

    // calculate ViewProjection matrix
    Matrix4 projection = Matrix4::perspectiveProjection(Rad{90.0f}, 4.0f / 3.0f, 0.1f, 100.f);

    
    // translate the world/view position,
    // note order of matrix multiply
    Matrix4 view = Matrix4::translation({0.0f, 0.0f, -30.0f}) *
    Matrix4::rotation(Deg{30.0f}, {1.0f, 0.0f, 0.0f}) *
    Matrix4::rotation(Deg{-22.5f*t}, {0.0f, 1.0f, 0.0f});

    shaderProgram.setProjMatrix(projection);
    shaderProgram.setViewMatrix(view);

    defaultFramebuffer.clear(FramebufferClear::Color | FramebufferClear::Depth);

    mesh.draw(shaderProgram);

    swapBuffers();

    redraw();
}

int main(int argc, char** argv) {
    BufferMapping app({argc, argv});
    return app.exec();
}
