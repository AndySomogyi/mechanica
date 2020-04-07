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


#pragma once

#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Shaders/Flat.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

using namespace Magnum;

using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;

class FlatShadeObject: public SceneGraph::Drawable3D {
    public:
        explicit FlatShadeObject(Object3D& object, Shaders::Flat3D& shader, const Color3& color, GL::Mesh& mesh, SceneGraph::DrawableGroup3D* const drawables): SceneGraph::Drawable3D{object, drawables}, _shader(shader), _color(color), _mesh(mesh) {}

        void draw(const Matrix4& transformation, SceneGraph::Camera3D& camera) override {
            _shader.setColor(_color)
                .setTransformationProjectionMatrix(camera.projectionMatrix() * transformation)
                .draw(_mesh);
        }

        FlatShadeObject& setColor(const Color3& color) { _color = color; return *this; }

    private:
        Shaders::Flat3D& _shader;
        Color3 _color;
        GL::Mesh& _mesh;
};


