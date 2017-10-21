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
#include <Magnum/Renderer.h>
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




        MxMesh *mesh;
        MxMeshRenderer renderer;
};

GmshTest1::GmshTest1(const Arguments& arguments):
    Platform::GlfwApplication{arguments,
        Configuration{}.setVersion(Version::GL410).
            setTitle("Gmsh Test 1")},
renderer{MxMeshRenderer::Flag::Wireframe},
mesh{new MxMesh()} {

    // need to enabler depth testing. The graphics processor can draw each facet in any order it wants.
    // Depth testing makes sure that front facing facts are drawn after back ones, so that back facets
    // don't cover up front ones.
    Renderer::enable(Renderer::Feature::DepthTest);

    // don't draw facets that face away from us. We have A LOT of these INSIDE cells, no need to
    // draw them.
    Renderer::enable(Renderer::Feature::FaceCulling);

    MxMeshGmshImporter importer{*mesh};

    importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");

    //importer.read("/Users/andy/src/mechanica/testing/gmsh1/sheet.msh");

    Vector3 min, max;
    std::tie(min, max) = mesh->extents();

    center = (max + min)/2;

    renderer.setMesh(mesh);

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

enum  ScalarTypes { S1, S2, S3 };

enum  VectorTypes {V1, V2, V3 };


int _type(ScalarTypes);
std::string _type(VectorTypes);

/*

template<typename T>
auto get(T val) -> decltype(_type(val));


template<>
auto get<ScalarTypes>(ScalarTypes val) -> decltype(_type(val)) { return 0;};

 */


struct foo {
    template<ScalarTypes s> static int get();

    template<VectorTypes s> static std::string get();
};

template<> int foo::get<ScalarTypes::S1>() {return 0; };


template<> std::string foo::get<VectorTypes::V1>() {return "V1"; };

//using VectorTypes;

namespace Stuff {
    enum Foo {F1, F2, F3};
}

void test();






void test() {
    int i = foo::get<ScalarTypes::S1>();

    std::string s = foo::get<V1>();

    //int z = foo::get<5>();

    int j = Stuff::F1;
}


int main(int argc, char** argv) {



    GmshTest1 app({argc, argv});
    return app.exec();
}


