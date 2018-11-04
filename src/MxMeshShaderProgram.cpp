#include "MxMeshShaderProgram.h"

#include <Corrade/Utility/Resource.h>

#include "Magnum/GL/Context.h"
#include "Magnum/GL/Extensions.h"
#include "Magnum/GL/Shader.h"
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
        MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL330);
        MAGNUM_ASSERT_GL_EXTENSION_SUPPORTED(GL::Extensions::ARB::geometry_shader4);
    }

    Utility::Resource rs("MxMeshShaderProgram");

    const GL::Version version = GL::Context::current().supportedVersion({GL::Version::GL330});
    CORRADE_INTERNAL_ASSERT(!flags || flags & Flag::NoGeometryShader || version >= GL::Version::GL330);

    GL::Shader vert = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Vertex);
    GL::Shader frag = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Fragment);

    vert.addSource(flags & Flag::Wireframe ? "#define WIREFRAME_RENDERING\n" : "")
        .addSource(flags & Flag::NoGeometryShader ? "#define NO_GEOMETRY_SHADER\n" : "")
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxMeshShaderProgram.vert"));
    
    std::cout << "Vertex Shader Source: " << std::endl;
    for (auto s : vert.sources()) {
        std::cout << s << std::endl;
    }

    frag.addSource(flags & Flag::Wireframe ? "#define WIREFRAME_RENDERING\n" : "")
        .addSource(flags & Flag::NoGeometryShader ? "#define NO_GEOMETRY_SHADER\n" : "")
        .addSource(rs.get("MxMeshShaderProgram.frag"));
    
    std::cout << "Fragment Shader Source: " << std::endl;
    for (auto s : frag.sources()) {
        std::cout << s << std::endl;
    }

    std::optional<GL::Shader> geom;
    if(flags & Flag::Wireframe && !(flags & Flag::NoGeometryShader)) {
        geom = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Geometry);
        geom->addSource(rs.get("MxMeshShaderProgram.geom"));
        
        std::cout << "Geometry Shader Source: " << std::endl;
        for (auto s : geom->sources()) {
            std::cout << s << std::endl;
        }
    }

    if(geom) {
        CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, *geom, frag}));
    }
    else {
        CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));
    }

    attachShaders({vert, frag});

    if(geom) attachShader(*geom);


    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
    {
        bindAttributeLocation(Position::Location, "position");

        bindAttributeLocation(Normal::Location, "normal");

        #if !defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_TARGET_GLES2)
        #ifndef MAGNUM_TARGET_GLES
        if(!GL::Context::current().isVersionSupported(GL::Version::GL310))
        #endif
        {
            bindAttributeLocation(VertexIndex::Location, "vertexIndex");
        }
        #endif
    }

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
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


