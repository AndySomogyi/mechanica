#include "MxMeshShaderProgram.h"

#include <Corrade/Utility/Resource.h>

#include "Magnum/Context.h"
#include "Magnum/Extensions.h"
#include "Magnum/Shader.h"
#include "MagnumExternal/Optional/optional.hpp"

#include "Magnum/Shaders/Implementation/CreateCompatibilityShader.h"

using namespace Magnum;
using namespace Magnum::Shaders;

int resourceInitializer_MxMeshShaderProgramRes();

void checkResource() {
    if (!Utility::Resource::hasGroup("MxMeshShaderProgram")) {
        resourceInitializer_MxMeshShaderProgramRes();
    }
}


MxMeshShaderProgram::MxMeshShaderProgram(const Flags flags): _flags{flags} {
    if(flags & Flag::Wireframe && !(flags & Flag::NoGeometryShader)) {
        MAGNUM_ASSERT_VERSION_SUPPORTED(Version::GL330);
        MAGNUM_ASSERT_EXTENSION_SUPPORTED(Extensions::GL::ARB::geometry_shader4);
    }

    Utility::Resource rs("MxMeshShaderProgram");

    const Version version = Context::current().supportedVersion({Version::GL330});
    CORRADE_INTERNAL_ASSERT(!flags || flags & Flag::NoGeometryShader || version >= Version::GL330);

    Shader vert = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, Shader::Type::Vertex);
    Shader frag = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, Shader::Type::Fragment);

    vert.addSource(flags & Flag::Wireframe ? "#define WIREFRAME_RENDERING\n" : "")
        .addSource(flags & Flag::NoGeometryShader ? "#define NO_GEOMETRY_SHADER\n" : "")
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxMeshShaderProgram.vert"));

    frag.addSource(flags & Flag::Wireframe ? "#define WIREFRAME_RENDERING\n" : "")
        .addSource(flags & Flag::NoGeometryShader ? "#define NO_GEOMETRY_SHADER\n" : "")
        .addSource(rs.get("MxMeshShaderProgram.frag"));

    std::optional<Shader> geom;
    if(flags & Flag::Wireframe && !(flags & Flag::NoGeometryShader)) {
        geom = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, Shader::Type::Geometry);
        geom->addSource(rs.get("MxMeshShaderProgram.geom"));
    }

    if(geom) {
        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, *geom, frag}));
    }
    else {
        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, frag}));
    }

    attachShaders({vert, frag});

    if(geom) attachShader(*geom);


    if(!Context::current().isExtensionSupported<Extensions::GL::ARB::explicit_attrib_location>(version))
    {
        bindAttributeLocation(Position::Location, "position");

        #if !defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_TARGET_GLES2)
        #ifndef MAGNUM_TARGET_GLES
        if(!Context::current().isVersionSupported(Version::GL310))
        #endif
        {
            bindAttributeLocation(VertexIndex::Location, "vertexIndex");
        }
        #endif
    }

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    if(!Context::current().isExtensionSupported<Extensions::GL::ARB::explicit_uniform_location>(version))
    {
        _transformationProjectionMatrixUniform = uniformLocation("transformationProjectionMatrix");
        _colorUniform = uniformLocation("color");
        if(flags & Flag::Wireframe) {
            _wireframeColorUniform = uniformLocation("wireframeColor");
            _wireframeWidthUniform = uniformLocation("wireframeWidth");
            _smoothnessUniform = uniformLocation("smoothness");
            if(!(flags & Flag::NoGeometryShader))
                _viewportSizeUniform = uniformLocation("viewportSize");
        }
    }
}


