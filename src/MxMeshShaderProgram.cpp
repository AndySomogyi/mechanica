#include "MxMeshShaderProgram.h"

#include <Corrade/Utility/Resource.h>

#include "Magnum/Context.h"
#include "Magnum/Extensions.h"
#include "Magnum/Shader.h"
#include "MagnumExternal/Optional/optional.hpp"

#include "Magnum/Shaders/Implementation/CreateCompatibilityShader.h"

//#include <iostream>

using namespace Magnum;
using namespace Magnum::Shaders;

int resourceInitializer_MxMeshShaderProgramRes();

void checkResource() {
    if (!Utility::Resource::hasGroup("MxMeshShaderProgram")) {
        resourceInitializer_MxMeshShaderProgramRes();
    }
}


MxMeshShaderProgram::MxMeshShaderProgram(const Flags flags): _flags{flags} {
    #ifndef MAGNUM_TARGET_GLES2
    if(flags & Flag::Wireframe && !(flags & Flag::NoGeometryShader)) {
        #ifndef MAGNUM_TARGET_GLES
        MAGNUM_ASSERT_VERSION_SUPPORTED(Version::GL320);
        MAGNUM_ASSERT_EXTENSION_SUPPORTED(Extensions::GL::ARB::geometry_shader4);
        #elif !defined(MAGNUM_TARGET_WEBGL)
        MAGNUM_ASSERT_EXTENSION_SUPPORTED(Extensions::GL::EXT::geometry_shader);
        #endif
    }
    #else
    if(_flags & Flag::Wireframe)
        MAGNUM_ASSERT_EXTENSION_SUPPORTED(Extensions::GL::OES::standard_derivatives);
    #endif


    Utility::Resource rs("MxMeshShaderProgram");

    #ifndef MAGNUM_TARGET_GLES
    const Version version = Context::current().supportedVersion({Version::GL320, Version::GL310, Version::GL300, Version::GL210});
    CORRADE_INTERNAL_ASSERT(!flags || flags & Flag::NoGeometryShader || version >= Version::GL320);
    #elif !defined(MAGNUM_TARGET_WEBGL)
    const Version version = Context::current().supportedVersion({Version::GLES310, Version::GLES300, Version::GLES200});
    CORRADE_INTERNAL_ASSERT(!flags || flags & Flag::NoGeometryShader || version >= Version::GLES310);
    #else
    const Version version = Context::current().supportedVersion({Version::GLES300, Version::GLES200});
    #endif

    Shader vert = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, Shader::Type::Vertex);
    Shader frag = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, Shader::Type::Fragment);

    vert.addSource(flags & Flag::Wireframe ? "#define WIREFRAME_RENDERING\n" : "")
        .addSource(flags & Flag::NoGeometryShader ? "#define NO_GEOMETRY_SHADER\n" : "")
        #ifdef MAGNUM_TARGET_WEBGL
        .addSource("#define SUBSCRIPTING_WORKAROUND\n")
        #elif defined(MAGNUM_TARGET_GLES2)
        .addSource(Context::current().detectedDriver() & Context::DetectedDriver::Angle ?
            "#define SUBSCRIPTING_WORKAROUND\n" : "")
        #endif
        .addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxMeshShaderProgram.vert"));
    frag.addSource(flags & Flag::Wireframe ? "#define WIREFRAME_RENDERING\n" : "")
        .addSource(flags & Flag::NoGeometryShader ? "#define NO_GEOMETRY_SHADER\n" : "")
        .addSource(rs.get("MxMeshShaderProgram.frag"));

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    std::optional<Shader> geom;
    if(flags & Flag::Wireframe && !(flags & Flag::NoGeometryShader)) {
        geom = Magnum::Shaders::Implementation::createCompatibilityShader(rs, version, Shader::Type::Geometry);
        geom->addSource(rs.get("MxMeshShaderProgram.geom"));
    }
    #endif

    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    if(geom) CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, *geom, frag}));
    else
    #endif
        CORRADE_INTERNAL_ASSERT_OUTPUT(Shader::compile({vert, frag}));

    attachShaders({vert, frag});
    #if !defined(MAGNUM_TARGET_GLES2) && !defined(MAGNUM_TARGET_WEBGL)
    if(geom) attachShader(*geom);
    #endif

    #ifndef MAGNUM_TARGET_GLES
    if(!Context::current().isExtensionSupported<Extensions::GL::ARB::explicit_attrib_location>(version))
    #else
    if(!Context::current().isVersionSupported(Version::GLES300))
    #endif
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

    #ifndef MAGNUM_TARGET_GLES
    if(!Context::current().isExtensionSupported<Extensions::GL::ARB::explicit_uniform_location>(version))
    #endif
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

    /* Set defaults in OpenGL ES (for desktop they are set in shader code itself) */
    #ifdef MAGNUM_TARGET_GLES
    setColor(Color3(1.0f));
    if(_flags & Flag::Wireframe) {
        setWireframeColor(Color3(0.0f));
        setWireframeWidth(1.0f);
        setSmoothness(2.0f);
    }
    #endif
}


