/*
 * SphereShader.cpp
 *
 *  Created on: Jul 6, 2017
 *      Author: andy
 */

#include "SphereShader.h"

SphereShader::SphereShader() {
    MAGNUM_ASSERT_VERSION_SUPPORTED(Version::GL330);

    Shader vert{Version::GL330, Shader::Type::Vertex};
    Shader geom{Version::GL330, Shader::Type::Geometry};
    Shader frag{Version::GL330, Shader::Type::Fragment};

    //vert.addSource(vertSrc);
    //geom.addSource(geomSrc);
    //frag.addSource(fragSrc);

    CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, geom, frag}));

    attachShaders({vert, geom, frag});

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    modelLoc = uniformLocation("model");
    viewLoc = uniformLocation("view");
    projLoc = uniformLocation("proj");
};


