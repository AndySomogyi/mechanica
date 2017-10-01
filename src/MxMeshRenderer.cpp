/*
 * MxMeshRenderer.cpp
 *
 *  Created on: Jul 7, 2017
 *      Author: andy
 */

#include <iostream>
#include <MxMeshRenderer.h>
#include <MxMeshShaderProgram.h>


//using namespace Magnum;





MxMeshRenderer::MxMeshRenderer(Flags flags) :
        shader{new MxMeshShaderProgram(flags)},
        mesh{nullptr} {
}

MxMeshRenderer& MxMeshRenderer::setViewportSize(const Magnum::Vector2& size) {
    shader->setViewportSize(size);
    return *this;
}

MxMeshRenderer& MxMeshRenderer::setColor(const Magnum::Color4& color) {
    shader->setColor(color);
    return *this;
}

MxMeshRenderer& MxMeshRenderer::setWireframeColor(const Magnum::Color4& color) {
    shader->setWireframeColor(color);
    return *this;
}

MxMeshRenderer& MxMeshRenderer::setWireframeWidth(float width) {
    shader->setWireframeWidth(width);
    return *this;
}

MxMeshRenderer& MxMeshRenderer::setSmoothness(float smoothness) {
    shader->setSmoothness(smoothness);
    return *this;
}

void dumpVertex(void* vertex, uint vertexCount) {

    Vector3 *p = (Vector3*)vertex;

    std::cout << "vertices: " << std::endl;
    std::cout << "---------" << std::endl;
    for (int i = 0; i < vertexCount; ++i) {
        std::cout << "{" << p[i][0] << "," << p[i][1] << "," << p[i][2] << "}" << std::endl;
        if ((i+1) % 3 == 0) {
            std::cout << "---------" << std::endl;

        }
    }
}

void dumpIndex(void* ind, uint indCount) {
    uint faceCnt = indCount / 3;
    uint *pint = (uint*)ind;

    std::cout << "indices: " << std::endl;
    for(int i = 0; i < faceCnt; ++i) {
        std::cout << "{" << pint[3*i] << "," << pint[3*i+1] << "," << pint[3*i+2] << "}" << std::endl;
    }

}




void MxMeshRenderer::draw() {
    for (int i = 0; i < gpuObjects.size(); ++i) {

        CellGPUObjects &obj = gpuObjects[i];
        MxCell& cell = *mesh->cells[i];

        //cell.dump();

        //uint* indexPtr = obj.indexBuffer.map<uint>(0,  cell.faceCount() * 3 * sizeof(uint),
        //    Buffer::MapFlag::Write|Buffer::MapFlag::Read);
        //cell.indexData(3* cell.faceCount(), indexPtr);

        //dumpIndex(indexPtr, cell.faceCount() * 3);


        obj.indexBuffer.unmap();

        void* vertexPtr = obj.vertexBuffers[0].map<void>(0,  cell.vertexCount() * sizeof(Vector3),
            Buffer::MapFlag::Write|Buffer::MapFlag::InvalidateBuffer);
        cell.vertexAtributeData({}, cell.vertexCount(), sizeof(Vector3), vertexPtr);

        //dumpVertex(vertexPtr, cell.vertexCount());

        obj.vertexBuffers[0].unmap();

        obj.mesh.draw(*shader);
    }
}

std::string glErrorString() {
    GLenum error = glGetError();
    switch (error) {
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_CONTEXT_LOST: return "GL_CONTEXT_LOST";
        default: return "who the fuck knows...";
    }
}

void MxMeshRenderer::addCellGPUObjects(MxCell& cell) {
    CellGPUObjects obj;

    Buffer vertexBuffer;
    vertexBuffer.setData({nullptr, cell.vertexCount() * sizeof(Vector3)}, BufferUsage::DynamicDraw);
    obj.vertexBuffers.push_back(std::move(vertexBuffer));


    //obj.indexBuffer.setData({nullptr, cell.faceCount() * 3 * sizeof(uint)}, BufferUsage::DynamicDraw);


    obj.mesh.setCount(cell.faceCount() * 3)
            .setPrimitive(MeshPrimitive::Triangles)
            .addVertexBuffer(obj.vertexBuffers[0], 0, Attribute<0, Vector3>{});
            //.setIndexBuffer(obj.indexBuffer, 0, Mesh::IndexType::UnsignedInt);

    gpuObjects.push_back(std::move(obj));
}

MxMeshRenderer::~MxMeshRenderer() {
    delete shader;
}

MxMeshRenderer& MxMeshRenderer::setMesh(MxMesh* _mesh) {
    mesh = _mesh;

    for(auto cell : mesh->cells) {
        addCellGPUObjects(*cell);
    }

    return *this;
}

MxMeshRenderer& MxMeshRenderer::setModelMatrix(const Magnum::Matrix4& mat) {
    modelMat = mat;
    shader->setTransformationProjectionMatrix(projMat * viewMat * modelMat);
    return *this;
}

MxMeshRenderer& MxMeshRenderer::setViewMatrix(const Magnum::Matrix4& mat) {
    viewMat = mat;
    shader->setTransformationProjectionMatrix(projMat * viewMat * modelMat);
    return *this;
}

MxMeshRenderer& MxMeshRenderer::setProjectionMatrix(
        const Magnum::Matrix4& mat) {
    projMat = mat;
    shader->setTransformationProjectionMatrix(projMat * viewMat * modelMat);
    return *this;
}



