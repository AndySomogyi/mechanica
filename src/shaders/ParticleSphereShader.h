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
#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/Math/Vector3.h>

using namespace Magnum;

class ParticleSphereShader: public GL::AbstractShaderProgram {
public:

    struct Vertex {
        Magnum::Vector3 pos;
        Magnum::Int index;
    };

    typedef Magnum::GL::Attribute<0, Magnum::Vector3> Position;

    typedef Magnum::GL::Attribute<1, Magnum::Int> Index;


    enum ColorMode {
        UniformDiffuseColor = 0,
        RampColorById,
        ConsistentRandom
    };

    explicit ParticleSphereShader();

    ParticleSphereShader& setNumParticles(Int numParticles);
    ParticleSphereShader& setParticleRadius(Float radius);

    ParticleSphereShader& setPointSizeScale(Float scale);
    ParticleSphereShader& setColorMode(Int colorMode);
    ParticleSphereShader& setAmbientColor(const Color3& color);
    ParticleSphereShader& setDiffuseColor(const Color3& color);
    ParticleSphereShader& setSpecularColor(const Color3& color);
    ParticleSphereShader& setShininess(Float shininess);

    ParticleSphereShader& setViewport(const Vector2i& viewport);
    ParticleSphereShader& setViewMatrix(const Matrix4& matrix);
    ParticleSphereShader& setProjectionMatrix(const Matrix4& matrix);
    ParticleSphereShader& setLightDirection(const Vector3& lightDir);

private:
    Int _uNumParticles,
    _uParticleRadius,
    _uPointSizeScale,
    _uColorMode,
    _uAmbientColor,
    _uDiffuseColor,
    _uSpecularColor,
    _uShininess,
    _uViewMatrix,
    _uProjectionMatrix,
    _uLightDir;
};


