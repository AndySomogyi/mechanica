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
#include <Magnum/Version.h>
#include <iostream>


#include <MxMeshGmshImporter.h>
#include <MxMeshRenderer.h>


#include <Magnum/Trade/MeshData3D.h>


using namespace std;
using namespace Magnum;
using namespace Magnum::Trade;
using namespace Magnum::Primitives;


class GmshTest1: public Platform::GlfwApplication {
    public:
        explicit GmshTest1(const Arguments& arguments);


private:
    void drawEvent() override;
    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;

    Matrix4 transformation, projection;
    Vector2i previousMousePosition;
    Color4 color;
    Vector3 center;




        MxMesh mesh;
        MxMeshRenderer renderer;
};

GmshTest1::GmshTest1(const Arguments& arguments):
    Platform::GlfwApplication{arguments,
        Configuration{}.setVersion(Version::GL410).
            setTitle("Voronoi Example")},
renderer{MxMeshRenderer::Flag::Wireframe} {

    MxMeshGmshImporter importer;

    mesh = importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");

    Vector3 min, max;
    std::tie(min, max) = mesh.extents();

    center = (max + min)/2;

    renderer.setMesh(mesh);

    //renderer.setModelMatrix(Matrix4::translation({0.0f, 0.0f, 1.0f}));

    //glDisable(GL_CULL_FACE​);


}



void GmshTest1::drawEvent() {
    defaultFramebuffer.clear(FramebufferClear::Color|FramebufferClear::Depth);

    renderer.setViewportSize(Vector2{defaultFramebuffer.viewport().size()});

    projection = Matrix4::perspectiveProjection(35.0_degf,
                                                Vector2{defaultFramebuffer.viewport().size()}.aspectRatio(), 0.01f, 100.0f);
    //* Matrix4::translation(Vector3::zAxis(-10.0f));


    renderer.setProjectionMatrix(projection);

    Matrix4 mat =   Matrix4::translation({0.0f, 0.0f, -5.0f}) * transformation  * Matrix4::translation(-center);

    renderer.setViewMatrix(mat);

    renderer.setColor(Color4::blue());

    renderer.setWireframeColor(Color4{0., 0., 0.});

    renderer.setWireframeWidth(0.5);

   // mesh.jiggle();

    renderer.draw();

    swapBuffers();

    redraw();

}

void GmshTest1::mousePressEvent(MouseEvent& event) {
    if(event.button() != MouseEvent::Button::Left) return;

    previousMousePosition = event.position();
    event.setAccepted();
}

void GmshTest1::mouseReleaseEvent(MouseEvent& event) {
    color = Color4::fromHsv(color.hue() + 50.0_degf, 1.0f, 1.0f);

    event.setAccepted();
    redraw();
}

void GmshTest1::mouseMoveEvent(MouseMoveEvent& event) {

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



    GmshTest1 app({argc, argv});
    return app.exec();
}


