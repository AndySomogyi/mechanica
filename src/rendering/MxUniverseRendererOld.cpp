/*
    This file is part of Mechanica.

    Based on Magnum example

    Original authors — credit is appreciated but not required:

        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 —
            Vladimír Vondruš <mosra@centrum.cz>
        2019 — Nghia Truong <nghiatruong.vn@gmail.com>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
 */


#include <Corrade/Utility/Assert.h>
#include <Corrade/Containers/ArrayView.h>
#include <rendering/MxUniverseRendererOld.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Functions.h>
#include <Magnum/Shaders/Generic.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Trade/MeshData3D.h>

#include <iostream>



using namespace Magnum::Math::Literals;

MxUniverseRendererOld::MxUniverseRendererOld(float particleRadius):
    _particleRadius(particleRadius),
    _mesh(GL::MeshPrimitive::Points) {
    _mesh.addVertexBuffer(_vertexBuffer, 0, ParticleSphereShader::Position{}, ParticleSphereShader::Index{});
    _shader.reset(new ParticleSphereShader);
}

MxUniverseRendererOld& MxUniverseRendererOld::draw(Containers::Pointer<SceneGraph::Camera3D>& camera,
        const Vector2i& viewportSize) {



    // the incomprehensible template madness way of doing things.
    //Containers::ArrayView<const float> data(reinterpret_cast<const float*>(&_points[0]), _points.size() * 3);
    //_bufferParticles.setData(data);


    // slightly more sensible way of doing things
    //_bufferParticles.setData({&_points[0], _points.size() * 3 * sizeof(float)},
    //                         GL::BufferUsage::DynamicDraw);

    // give me the damned bytes...



    // invalidate / resize the buffer
    _vertexBuffer.setData({NULL, _Engine.s.nr_parts * sizeof(ParticleSphereShader::Vertex)},
            GL::BufferUsage::DynamicDraw);

    // get pointer to data
    void* tmp = _vertexBuffer.map(0,
            _Engine.s.nr_parts * sizeof(ParticleSphereShader::Vertex),
            GL::Buffer::MapFlag::Write|GL::Buffer::MapFlag::InvalidateBuffer);

    ParticleSphereShader::Vertex* vertexPtr = (ParticleSphereShader::Vertex*)tmp;


    int i = 0;
    for (int cid = 0 ; cid < _Engine.s.nr_cells ; cid++ ) {
        for (int pid = 0 ; pid < _Engine.s.cells[cid].count ; pid++ ) {
            vertexPtr[i].pos.x() = _Engine.s.cells[cid].origin[0] + _Engine.s.cells[cid].parts[pid].x[0];
            vertexPtr[i].pos.y() = _Engine.s.cells[cid].origin[1] + _Engine.s.cells[cid].parts[pid].x[1];
            vertexPtr[i].pos.z() = _Engine.s.cells[cid].origin[2] + _Engine.s.cells[cid].parts[pid].x[2];
            vertexPtr[i].index = _Engine.s.cells[cid].parts[pid].id;

            MxParticle *p  = &_Engine.s.cells[cid].parts[pid];
            i++;
        }
    }

    _vertexBuffer.unmap();


    _mesh.setCount(_Engine.s.nr_parts);
    _dirty = false;


    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    (*_shader)
                /* particle data */
                .setNumParticles(_Engine.s.nr_parts)
                .setParticleRadius(_particleRadius)
                /* sphere render data */
                .setPointSizeScale(static_cast<float>(viewportSize.x())/
                        Math::tan(22.5_degf)) /* tan(half field-of-view angle (45_deg)*/
                        .setColorMode(_colorMode)
                        .setAmbientColor(_ambientColor)
                        .setDiffuseColor(_diffuseColor)
                        .setSpecularColor(_specularColor)
                        .setShininess(_shininess)
                        /* view/prj matrices and light */
                        .setViewMatrix(camera->cameraMatrix() * modelViewMat)
                        .setProjectionMatrix(camera->projectionMatrix())
                        .setLightDirection(_lightDir)
                        .draw(_mesh);



    return *this;
}



