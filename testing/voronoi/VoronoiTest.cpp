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
#include <MagnumPlugins/AssimpImporter/AssimpImporter.h>
#include <Magnum/Version.h>
#include <iostream>

#include "VoronoiTesselator.h"

#include <MxMeshVoronoiImporter.h>
#include <MxMeshRenderer.h>


#include <Magnum/Trade/MeshData3D.h>


using namespace std;
using namespace Magnum;
using namespace Magnum::Trade;
using namespace Magnum::Primitives;

typedef std::optional<MeshData3D> OptMeshData3D;

OptMeshData3D  read_points(int argc, char** argv);

OptMeshData3D  read_points(int argc, char** argv) {

    /* Load scene importer plugin */
    PluginManager::Manager<Trade::AbstractImporter> manager{};
    std::unique_ptr<Trade::AbstractImporter> importer = manager.loadAndInstantiate("ObjImporter");

    std::ignore = argc;

    if (importer->openFile(argv[1])) {
        cout << "opened file \"" << argv[1] << "\" OK" << endl;
        cout << "mesh 3d count: " << importer->mesh3DCount() << std::endl;
    } else {
        cout << "failed to open " <<  argv[1] << endl;
        return OptMeshData3D{};
    }

    int defScene = importer->defaultScene();
    cout << "default scene: " << defScene;

    return importer->mesh3D(defScene);
}

class VoronoiTest: public Platform::GlfwApplication {
    public:
        explicit VoronoiTest(const Arguments& arguments);


private:
    void drawEvent() override;
    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;
    
    Matrix4 transformation, projection;
    Vector2i previousMousePosition;
    Color4 color;




        MxMesh mesh;
        MxMeshRenderer renderer;
};

VoronoiTest::VoronoiTest(const Arguments& arguments):
    Platform::GlfwApplication{arguments,
        Configuration{}.setVersion(Version::GL410).
            setTitle("Voronoi Example")},
renderer{MxMeshRenderer::Flag::Wireframe} {
                

    //OptMeshData3D points = read_points(arguments.argc, arguments.argv);

    //auto result = VoronoiTesselator::tesselate(points->positions(0),
    //            {0,0,0}, {1,1,1}, {1,1,1});



    MxMeshVoronoiImporter::readFile("/Users/andy/src/mechanica/testing/voronoi/points.obj",
            {0,0,0}, {10,10,10}, {10,10,10}, {{false, false, false}}, mesh);

    renderer.setMesh(mesh);
                
    //renderer.setModelMatrix(Matrix4::translation({0.0f, 0.0f, 1.0f}));
                
    //glDisable(GL_CULL_FACE​);


}



void VoronoiTest::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color|FramebufferClear::Depth);
    
    
    
    renderer.setViewportSize(Vector2{defaultFramebuffer.viewport().size()});
    
    projection = Matrix4::perspectiveProjection(35.0_degf,
                                                Vector2{defaultFramebuffer.viewport().size()}.aspectRatio(), 0.01f, 100.0f);
    //* Matrix4::translation(Vector3::zAxis(-10.0f));

    
    renderer.setProjectionMatrix(projection);
    
    Matrix4 mat =   Matrix4::translation({-1.0f, 0.5f, -40.0f}) * transformation  * Matrix4::translation({-2.0f, -5.f, -5.f});
    
    renderer.setViewMatrix(mat);
    
    renderer.setColor(Color4::blue());
    
    renderer.setWireframeColor(Color4{0., 0., 0.});
    
    renderer.setWireframeWidth(0.5);
    
    renderer.draw();
    
    swapBuffers();
}

void VoronoiTest::mousePressEvent(MouseEvent& event) {
    if(event.button() != MouseEvent::Button::Left) return;
    
    previousMousePosition = event.position();
    event.setAccepted();
}

void VoronoiTest::mouseReleaseEvent(MouseEvent& event) {
    color = Color4::fromHsv(color.hue() + 50.0_degf, 1.0f, 1.0f);
    
    event.setAccepted();
    redraw();
}

void VoronoiTest::mouseMoveEvent(MouseMoveEvent& event) {
    
    if(glfwGetMouseButton(window(), GLFW_MOUSE_BUTTON_1) != GLFW_PRESS) return;
    
    const Vector2 delta = 3.0f*
    Vector2{event.position() - previousMousePosition}/
    Vector2{defaultFramebuffer.viewport().size()};
    
    transformation =
    
    Matrix4::rotationX(Rad{delta.y()})*
    transformation*
    Matrix4::rotationY(Rad{delta.x()});
    
    previousMousePosition = event.position();
    event.setAccepted();
    redraw();
}


int main(int argc, char** argv) {

    CORRADE_PLUGIN_IMPORT(AnyImageImporter);
    CORRADE_PLUGIN_IMPORT(ObjImporter);
    CORRADE_PLUGIN_IMPORT(AssimpImporter);
    CORRADE_PLUGIN_IMPORT(OpenGexImporter);
    CORRADE_PLUGIN_IMPORT(AnySceneImporter);

    VoronoiTest app({argc, argv});
    return app.exec();
}


