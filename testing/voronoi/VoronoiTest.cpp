/**
 * Testing of voronoi tesselation, hook up Voro++ with Magnum and Assimp to read points into a 
 * MeshData3D struct, perform a voronoi tesselation to genenerate a mesh, and use Magnum
 * to display that mesh. 
 */

#include <Magnum/Buffer.h>
#include <Magnum/DefaultFramebuffer.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Vector3.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Shaders/VertexColor.h>
#include <Magnum/Primitives/Cube.h>

#include <iostream>
#include "AssimpImporter.h"

#include <Magnum/Trade/MeshData3D.h>


using namespace std;
using namespace Magnum;
using namespace Magnum::Trade;
using namespace Magnum::Primitives;

std::optional<MeshData3D>  read_points(int argc, char** argv);

std::optional<MeshData3D>  read_points(int argc, char** argv) {

    std::ignore = argc;

    AssimpImporter assimp;

    if (assimp.openFile(argv[1])) {
        cout << "opened file \"" << argv[1] << "\" OK" << endl;
        cout << "mesh 3d count: " << assimp.mesh3DCount() << std::endl;
    } else {
        cout << "failed to open " <<  argv[1] << endl;
    }

    return  Cube::solid();
}

class VoronoiTest: public Platform::Application {
    public:
        explicit VoronoiTest(const Arguments& arguments);

    private:
        void drawEvent() override;

        Buffer _buffer;
        Mesh _mesh;
        Shaders::VertexColor2D _shader;
};

VoronoiTest::VoronoiTest(const Arguments& arguments):
    Platform::Application{arguments, Configuration{}.setTitle("Magnum Triangle Example")} {
    using namespace Math::Literals;

    read_points(arguments.argc, arguments.argv);

    struct TriangleVertex {
        Vector2 position;
        Color3 color;
    };
    static const TriangleVertex data[]{
        {{-0.5f, -0.5f}, 0xff0000_srgbf},   /* Left vertex, red color */
        {{ 0.5f, -0.5f}, 0x00ff00_srgbf},   /* Right vertex, green color */
        {{ 0.0f,  0.5f}, 0x0000ff_srgbf}    /* Top vertex, blue color */
    };

    _buffer.setData(data, BufferUsage::StaticDraw);
    _mesh.setPrimitive(MeshPrimitive::Triangles)
        .setCount(3)
        .addVertexBuffer(_buffer, 0,
            Shaders::VertexColor2D::Position{},
            Shaders::VertexColor2D::Color{Shaders::VertexColor2D::Color::Components::Three});
}

void VoronoiTest::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color);

    _mesh.draw(_shader);

    swapBuffers();
}

int main(int argc, char** argv) {
    VoronoiTest app({argc, argv});
    return app.exec();
}


