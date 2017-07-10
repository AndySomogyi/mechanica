/*
 * SphereShader.h
 *
 *  Created on: Jul 6, 2017
 *      Author: andy
 */

#ifndef TESTING_VORONOI_SPHERESHADER_H_
#define TESTING_VORONOI_SPHERESHADER_H_

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

using namespace Magnum;

/**
 * Renders a set of spheres for a given set of positions, sizes and colors.
 *
 * Performs indexed rendering, where each vertex in the set of positions
 * is used as the center of a sphere to draw.
 */

class SphereShader : public AbstractShaderProgram {

public:

    typedef Attribute<0, Vector3> PosAttr;
    typedef Attribute<1, Color3> ColorAttr;
    typedef Attribute<2, float> SizeAttr;

    explicit SphereShader();


    void setModelMatrix(const Matrix4& mat) {
        setUniform(modelLoc, mat);
    }

    void setViewMatrix(const Matrix4& mat) {
        setUniform(viewLoc, mat);
    }

    void setProjMatrix(const Matrix4& mat) {
        setUniform(projLoc, mat);
    }

private:
    int modelLoc, viewLoc, projLoc;
};

#endif /* TESTING_VORONOI_SPHERESHADER_H_ */
