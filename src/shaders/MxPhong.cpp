/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                2020, 2021 Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include "MxPhong.h"

#if defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_BUILD_DEPRECATED)
#include <Corrade/Containers/Array.h>
#endif
#include <Corrade/Containers/EnumSet.hpp>
#include <Corrade/Containers/Reference.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/Utility/FormatStl.h>
#include <Corrade/Utility/Resource.h>

#include "Magnum/GL/Context.h"
#include "Magnum/GL/Extensions.h"
#include "Magnum/GL/Shader.h"
#include "Magnum/GL/Texture.h"
#include "Magnum/Math/Color.h"
#include "Magnum/Math/Matrix3.h"
#include "Magnum/Math/Matrix4.h"

#include <CLogger.hpp>

#include <iostream>

#include "Magnum/Shaders/Implementation/CreateCompatibilityShader.h"

namespace Magnum { namespace Shaders {

namespace {
    enum: Int {
        AmbientTextureUnit = 0,
        DiffuseTextureUnit = 1,
        SpecularTextureUnit = 2,
        NormalTextureUnit = 3
    };
}

MxPhong::MxPhong(Flags flags, unsigned lightCount, unsigned clipPlaneCount):
                 _flags{flags},
                 _lightCount{lightCount},
                 _clipPlaneCount{clipPlaneCount},
                 _lightColorsUniform{_lightPositionsUniform + Int(lightCount)},
                 _lightSpecularColorsUniform{_lightPositionsUniform + 2*Int(lightCount)},
                 _lightRangesUniform{_lightPositionsUniform + 3*Int(lightCount)},
                 _clipPlanesUniform{_lightPositionsUniform + 4*Int(lightCount)} {
                     
    CORRADE_ASSERT(!(flags & Flag::TextureTransformation) ||
                   (flags & (Flag::AmbientTexture|Flag::DiffuseTexture|Flag::SpecularTexture|Flag::NormalTexture)),
                   "Shaders::Phong: texture transformation enabled but the shader is not textured", );

    Utility::Resource rs("MxMeshShaderProgram");

    #ifndef MAGNUM_TARGET_GLES
    const GL::Version version = GL::Context::current().supportedVersion({GL::Version::GL320, GL::Version::GL310, GL::Version::GL300, GL::Version::GL210});
    #else
    const GL::Version version = GL::Context::current().supportedVersion({GL::Version::GLES300, GL::Version::GLES200});
    #endif

    GL::Shader vert = Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Vertex);
    GL::Shader frag = Implementation::createCompatibilityShader(rs, version, GL::Shader::Type::Fragment);

    #ifndef MAGNUM_TARGET_GLES
    std::string lightInitializerVertex, lightInitializerFragment;
    if(lightCount) {
        using namespace Containers::Literals;

        /* Initializer for the light color / position / range arrays -- we need
           a list of initializers joined by commas. For GLES we'll simply
           upload the values directly. */
        constexpr Containers::StringView lightPositionInitializerPreamble = "#define LIGHT_POSITION_INITIALIZER "_s;
        constexpr Containers::StringView lightColorInitializerPreamble = "#define LIGHT_COLOR_INITIALIZER "_s;
        constexpr Containers::StringView lightRangeInitializerPreamble = "#define LIGHT_RANGE_INITIALIZER "_s;
        constexpr Containers::StringView lightPositionInitializerItem = "vec4(0.0, 0.0, 1.0, 0.0), "_s;
        constexpr Containers::StringView lightColorInitializerItem = "vec3(1.0), "_s;
        constexpr Containers::StringView lightRangeInitializerItem = "1.0/0.0, "_s;

        lightInitializerVertex.reserve(
            lightPositionInitializerPreamble.size() +
            lightCount*(lightPositionInitializerItem.size()));

        lightInitializerVertex.append(lightPositionInitializerPreamble.data(), lightPositionInitializerPreamble.size());
        for(std::size_t i = 0; i != lightCount; ++i)
            lightInitializerVertex.append(lightPositionInitializerItem.data(), lightPositionInitializerItem.size());

        /* Drop the last comma and add a newline at the end */
        lightInitializerVertex[lightInitializerVertex.size() - 2] = '\n';
        lightInitializerVertex.resize(lightInitializerVertex.size() - 1);

        lightInitializerFragment.reserve(
            lightColorInitializerPreamble.size() +
            lightRangeInitializerPreamble.size() +
            lightCount*(lightColorInitializerItem.size() +
                        lightRangeInitializerItem.size()));

        lightInitializerFragment.append(lightColorInitializerPreamble.data(), lightColorInitializerPreamble.size());
        for(std::size_t i = 0; i != lightCount; ++i)
            lightInitializerFragment.append(lightColorInitializerItem.data(), lightColorInitializerItem.size());

        /* Drop the last comma and add a newline at the end */
        lightInitializerFragment[lightInitializerFragment.size() - 2] = '\n';
        lightInitializerFragment.resize(lightInitializerFragment.size() - 1);

        lightInitializerFragment.append(lightRangeInitializerPreamble.data(), lightRangeInitializerPreamble.size());
        for(std::size_t i = 0; i != lightCount; ++i)
            lightInitializerFragment.append(lightRangeInitializerItem.data(), lightRangeInitializerItem.size());

        /* Drop the last comma and add a newline at the end */
        lightInitializerFragment[lightInitializerFragment.size() - 2] = '\n';
        lightInitializerFragment.resize(lightInitializerFragment.size() - 1);
    }
    #endif

    vert.addSource(flags & (Flag::AmbientTexture|Flag::DiffuseTexture|Flag::SpecularTexture|Flag::NormalTexture) ? "#define TEXTURED\n" : "")
        .addSource(flags & Flag::NormalTexture ? "#define NORMAL_TEXTURE\n" : "")
        .addSource(flags & Flag::Bitangent ? "#define BITANGENT\n" : "")
        .addSource(flags & Flag::VertexColor ? "#define VERTEX_COLOR\n" : "")
        .addSource(flags & Flag::TextureTransformation ? "#define TEXTURE_TRANSFORMATION\n" : "")
        .addSource(Utility::formatString("#define LIGHT_COUNT {}\n", lightCount))
        .addSource(Utility::formatString("#define CLIP_PLANE_COUNT {}\n", clipPlaneCount))
        .addSource(Utility::formatString("#define CLIP_PLANES_UNIFORM {}\n", _clipPlanesUniform))
        #ifndef MAGNUM_TARGET_GLES2
        .addSource(flags >= Flag::InstancedObjectId ? "#define INSTANCED_OBJECT_ID\n" : "")
        #endif
        .addSource(flags & Flag::InstancedTransformation ? "#define INSTANCED_TRANSFORMATION\n" : "")
        .addSource(flags >= Flag::InstancedTextureOffset ? "#define INSTANCED_TEXTURE_OFFSET\n" : "");
    #ifndef MAGNUM_TARGET_GLES
    if(lightCount) vert.addSource(std::move(lightInitializerVertex));
    #endif
    vert.addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxPhong.vert"));
    frag.addSource(flags & Flag::AmbientTexture ? "#define AMBIENT_TEXTURE\n" : "")
        .addSource(flags & Flag::DiffuseTexture ? "#define DIFFUSE_TEXTURE\n" : "")
        .addSource(flags & Flag::SpecularTexture ? "#define SPECULAR_TEXTURE\n" : "")
        .addSource(flags & Flag::NormalTexture ? "#define NORMAL_TEXTURE\n" : "")
        .addSource(flags & Flag::Bitangent ? "#define BITANGENT\n" : "")
        .addSource(flags & Flag::VertexColor ? "#define VERTEX_COLOR\n" : "")
        .addSource(flags & Flag::AlphaMask ? "#define ALPHA_MASK\n" : "")
        #ifndef MAGNUM_TARGET_GLES2
        .addSource(flags & Flag::ObjectId ? "#define OBJECT_ID\n" : "")
        .addSource(flags >= Flag::InstancedObjectId ? "#define INSTANCED_OBJECT_ID\n" : "")
        #endif
        .addSource(Utility::formatString(
            "#define LIGHT_COUNT {}\n"
            "#define LIGHT_COLORS_LOCATION {}\n"
            "#define LIGHT_SPECULAR_COLORS_LOCATION {}\n"
            "#define LIGHT_RANGES_LOCATION {}\n",
            lightCount,
            _lightPositionsUniform + lightCount,
            _lightPositionsUniform + 2*lightCount,
            _lightPositionsUniform + 3*lightCount));
                     
    #ifndef MAGNUM_TARGET_GLES
    if(lightCount) frag.addSource(std::move(lightInitializerFragment));
    #endif
    frag.addSource(rs.get("generic.glsl"))
        .addSource(rs.get("MxPhong.frag"));
          
                     
     if(CLogger::getLevel() >= LOG_DEBUG) {
         std::stringstream ss;
         
         for(auto s : vert.sources()) {
             ss << s << std::endl;
         }
         
         Log(LOG_DEBUG) << "creating vertex shader: " << std::endl << ss.str();
         
         ss.clear();
         
         
         for(auto s : frag.sources()) {
             ss << s << std::endl;
         }
         
         Log(LOG_DEBUG) << "creating fragment shader: " << std::endl << ss.str();
     }

    CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

    attachShaders({vert, frag});

    /* ES3 has this done in the shader directly and doesn't even provide
       bindFragmentDataLocation() */
    #if !defined(MAGNUM_TARGET_GLES) || defined(MAGNUM_TARGET_GLES2)
    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_attrib_location>(version))
    #endif
    {
        bindAttributeLocation(Position::Location, "position");
        if(lightCount)
            bindAttributeLocation(Normal::Location, "normal");
        if((flags & Flag::NormalTexture) && lightCount) {
            bindAttributeLocation(Tangent::Location, "tangent");
            if(flags & Flag::Bitangent)
                bindAttributeLocation(Bitangent::Location, "bitangent");
        }
        if(flags & Flag::VertexColor)
            bindAttributeLocation(Color3::Location, "vertexColor"); /* Color4 is the same */
        if(flags & (Flag::AmbientTexture|Flag::DiffuseTexture|Flag::SpecularTexture))
            bindAttributeLocation(TextureCoordinates::Location, "textureCoordinates");
        #ifndef MAGNUM_TARGET_GLES2
        if(flags & Flag::ObjectId) {
            bindFragmentDataLocation(ColorOutput, "color");
            bindFragmentDataLocation(ObjectIdOutput, "objectId");
        }
        if(flags >= Flag::InstancedObjectId)
            bindAttributeLocation(ObjectId::Location, "instanceObjectId");
        #endif
        if(flags & Flag::InstancedTransformation)
            bindAttributeLocation(TransformationMatrix::Location, "instancedTransformationMatrix");
        if(flags >= Flag::InstancedTextureOffset)
            bindAttributeLocation(TextureOffset::Location, "instancedTextureOffset");
    }
    #endif

    CORRADE_INTERNAL_ASSERT_OUTPUT(link());

    #ifndef MAGNUM_TARGET_GLES
    if(!GL::Context::current().isExtensionSupported<GL::Extensions::ARB::explicit_uniform_location>(version))
    #endif
    {
        _transformationMatrixUniform = uniformLocation("transformationMatrix");
        if(flags & Flag::TextureTransformation)
            _textureMatrixUniform = uniformLocation("textureMatrix");
        _projectionMatrixUniform = uniformLocation("projectionMatrix");
        _ambientColorUniform = uniformLocation("ambientColor");
        if(lightCount) {
            _normalMatrixUniform = uniformLocation("normalMatrix");
            _diffuseColorUniform = uniformLocation("diffuseColor");
            _specularColorUniform = uniformLocation("specularColor");
            _shininessUniform = uniformLocation("shininess");
            if(flags & Flag::NormalTexture)
                _normalTextureScaleUniform = uniformLocation("normalTextureScale");
            _lightPositionsUniform = uniformLocation("lightPositions");
            _lightColorsUniform = uniformLocation("lightColors");
            _lightSpecularColorsUniform = uniformLocation("lightSpecularColors");
            _lightRangesUniform = uniformLocation("lightRanges");
        }
        
        if(clipPlaneCount) {
            _clipPlanesUniform = uniformLocation("clipPlanes");
        }
        
        if(flags & Flag::AlphaMask) _alphaMaskUniform = uniformLocation("alphaMask");
        #ifndef MAGNUM_TARGET_GLES2
        if(flags & Flag::ObjectId) _objectIdUniform = uniformLocation("objectId");
        #endif
    }

    #ifndef MAGNUM_TARGET_GLES
    if(flags && !GL::Context::current().isExtensionSupported<GL::Extensions::ARB::shading_language_420pack>(version))
    #endif
    {
        if(flags & Flag::AmbientTexture) setUniform(uniformLocation("ambientTexture"), AmbientTextureUnit);
        if(lightCount) {
            if(flags & Flag::DiffuseTexture) setUniform(uniformLocation("diffuseTexture"), DiffuseTextureUnit);
            if(flags & Flag::SpecularTexture) setUniform(uniformLocation("specularTexture"), SpecularTextureUnit);
            if(flags & Flag::NormalTexture) setUniform(uniformLocation("normalTexture"), NormalTextureUnit);
        }
    }

    /* Set defaults in OpenGL ES (for desktop they are set in shader code itself) */
    #ifdef MAGNUM_TARGET_GLES
    /* Default to fully opaque white so we can see the textures */
    if(flags & Flag::AmbientTexture) setAmbientColor(Magnum::Color4{1.0f});
    else setAmbientColor(Magnum::Color4{0.0f});
    setTransformationMatrix(Matrix4{Math::IdentityInit});
    setProjectionMatrix(Matrix4{Math::IdentityInit});
    if(lightCount) {
        setDiffuseColor(Magnum::Color4{1.0f});
        setSpecularColor(Magnum::Color4{1.0f, 0.0f});
        setShininess(80.0f);
        if(flags & Flag::NormalTexture)
            setNormalTextureScale(1.0f);
        setLightPositions(Containers::Array<Vector4>{Containers::DirectInit, lightCount, Vector4{0.0f, 0.0f, 1.0f, 0.0f}});
        Containers::Array<Magnum::Color3> colors{Containers::DirectInit, lightCount, Magnum::Color3{1.0f}};
        setLightColors(colors);
        setLightSpecularColors(colors);
        setLightRanges(Containers::Array<Float>{Containers::DirectInit, lightCount, Constants::inf()});
        /* Light position is zero by default */
        setNormalMatrix(Matrix3x3{Math::IdentityInit});
    }
    if(flags & Flag::TextureTransformation)
        setTextureMatrix(Matrix3{Math::IdentityInit});
    if(flags & Flag::AlphaMask) setAlphaMask(0.5f);
    /* Object ID is zero by default */
    #endif
}

MxPhong& MxPhong::setAmbientColor(const Magnum::Color4& color) {
    setUniform(_ambientColorUniform, color);
    return *this;
}

MxPhong& MxPhong::bindAmbientTexture(GL::Texture2D& texture) {
    CORRADE_ASSERT(_flags & Flag::AmbientTexture,
        "Shaders::MxPhong::bindAmbientTexture(): the shader was not created with ambient texture enabled", *this);
    texture.bind(AmbientTextureUnit);
    return *this;
}

MxPhong& MxPhong::setDiffuseColor(const Magnum::Color4& color) {
    if(_lightCount) setUniform(_diffuseColorUniform, color);
    return *this;
}

MxPhong& MxPhong::bindDiffuseTexture(GL::Texture2D& texture) {
    CORRADE_ASSERT(_flags & Flag::DiffuseTexture,
        "Shaders::MxPhong::bindDiffuseTexture(): the shader was not created with diffuse texture enabled", *this);
    if(_lightCount) texture.bind(DiffuseTextureUnit);
    return *this;
}

MxPhong& MxPhong::setSpecularColor(const Magnum::Color4& color) {
    if(_lightCount) setUniform(_specularColorUniform, color);
    return *this;
}

MxPhong& MxPhong::bindSpecularTexture(GL::Texture2D& texture) {
    CORRADE_ASSERT(_flags & Flag::SpecularTexture,
        "Shaders::MxPhong::bindSpecularTexture(): the shader was not created with specular texture enabled", *this);
    if(_lightCount) texture.bind(SpecularTextureUnit);
    return *this;
}

MxPhong& MxPhong::bindNormalTexture(GL::Texture2D& texture) {
    CORRADE_ASSERT(_flags & Flag::NormalTexture,
        "Shaders::MxPhong::bindNormalTexture(): the shader was not created with normal texture enabled", *this);
    if(_lightCount) texture.bind(NormalTextureUnit);
    return *this;
}

MxPhong& MxPhong::bindTextures(GL::Texture2D* ambient, GL::Texture2D* diffuse, GL::Texture2D* specular, GL::Texture2D* normal) {
    CORRADE_ASSERT(_flags & (Flag::AmbientTexture|Flag::DiffuseTexture|Flag::SpecularTexture|Flag::NormalTexture),
        "Shaders::MxPhong::bindTextures(): the shader was not created with any textures enabled", *this);
    GL::AbstractTexture::bind(AmbientTextureUnit, {ambient, diffuse, specular, normal});
    return *this;
}

MxPhong& MxPhong::setShininess(Float shininess) {
    if(_lightCount) setUniform(_shininessUniform, shininess);
    return *this;
}

MxPhong& MxPhong::setNormalTextureScale(const Float scale) {
    CORRADE_ASSERT(_flags & Flag::NormalTexture,
        "Shaders::MxPhong::setNormalTextureScale(): the shader was not created with normal texture enabled", *this);
    if(_lightCount) setUniform(_normalTextureScaleUniform, scale);
    return *this;
}

MxPhong& MxPhong::setAlphaMask(Float mask) {
    CORRADE_ASSERT(_flags & Flag::AlphaMask,
        "Shaders::MxPhong::setAlphaMask(): the shader was not created with alpha mask enabled", *this);
    setUniform(_alphaMaskUniform, mask);
    return *this;
}

#ifndef MAGNUM_TARGET_GLES2
MxPhong& MxPhong::setObjectId(UnsignedInt id) {
    CORRADE_ASSERT(_flags & Flag::ObjectId,
        "Shaders::MxPhong::setObjectId(): the shader was not created with object ID enabled", *this);
    setUniform(_objectIdUniform, id);
    return *this;
}
#endif

MxPhong& MxPhong::setTransformationMatrix(const Matrix4& matrix) {
    setUniform(_transformationMatrixUniform, matrix);
    return *this;
}

MxPhong& MxPhong::setNormalMatrix(const Matrix3x3& matrix) {
    if(_lightCount) setUniform(_normalMatrixUniform, matrix);
    return *this;
}

MxPhong& MxPhong::setProjectionMatrix(const Matrix4& matrix) {
    setUniform(_projectionMatrixUniform, matrix);
    return *this;
}

MxPhong& MxPhong::setTextureMatrix(const Matrix3& matrix) {
    CORRADE_ASSERT(_flags & Flag::TextureTransformation,
        "Shaders::MxPhong::setTextureMatrix(): the shader was not created with texture transformation enabled", *this);
    setUniform(_textureMatrixUniform, matrix);
    return *this;
}

MxPhong& MxPhong::setLightPositions(const Containers::ArrayView<const Vector4> positions) {
    CORRADE_ASSERT(_lightCount == positions.size(),
        "Shaders::MxPhong::setLightPositions(): expected" << _lightCount << "items but got" << positions.size(), *this);
    if(_lightCount) setUniform(_lightPositionsUniform, positions);
    return *this;
}

/* It's light, but can't be in the header because MSVC needs to know the size
   of Vector3 for the initializer list use */
MxPhong& MxPhong::setLightPositions(const std::initializer_list<Vector4> positions) {
    return setLightPositions(Containers::arrayView(positions));
}

#ifdef MAGNUM_BUILD_DEPRECATED
MxPhong& MxPhong::setLightPositions(const Containers::ArrayView<const Vector3> positions) {
    Containers::Array<Vector4> fourComponent{Containers::NoInit, positions.size()};
    for(std::size_t i = 0; i != positions.size(); ++i)
        fourComponent[i] = Vector4{positions[i], 0.0f};
    setLightPositions(fourComponent);
    return *this;
}

MxPhong& MxPhong::setLightPositions(const std::initializer_list<Vector3> positions) {
    CORRADE_IGNORE_DEPRECATED_PUSH
    return setLightPositions(Containers::arrayView(positions));
    CORRADE_IGNORE_DEPRECATED_POP
}
#endif

MxPhong& MxPhong::setLightPosition(const UnsignedInt id, const Vector4& position) {
    CORRADE_ASSERT(id < _lightCount,
        "Shaders::MxPhong::setLightPosition(): light ID" << id << "is out of bounds for" << _lightCount << "lights", *this);
    setUniform(_lightPositionsUniform + id, position);
    return *this;
}


MxPhong& MxPhong::setclipPlaneEquation(const UnsignedInt id, const Vector4& planeEquation) {
    CORRADE_ASSERT(id < _clipPlaneCount,
        "Shaders::MxPhong::setclipPlaneEquation(): plane ID" << id << "is out of bounds for" << _clipPlaneCount << "planes", *this);
    setUniform(_clipPlanesUniform + id, planeEquation);
    return *this;
}

#ifdef MAGNUM_BUILD_DEPRECATED
MxPhong& MxPhong::setLightPosition(UnsignedInt id, const Vector3& position) {
    return setLightPosition(id, Vector4{position, 0.0f});
}

MxPhong& MxPhong::setLightPosition(const Vector3& position) {
    /* Use the list variant to check the shader really has just one light */
    return setLightPositions({Vector4{position, 0.0f}});
}
#endif

MxPhong& MxPhong::setLightColors(const Containers::ArrayView<const Magnum::Color3> colors) {
    CORRADE_ASSERT(_lightCount == colors.size(),
        "Shaders::MxPhong::setLightColors(): expected" << _lightCount << "items but got" << colors.size(), *this);
    if(_lightCount) setUniform(_lightColorsUniform, colors);
    return *this;
}

#ifdef MAGNUM_BUILD_DEPRECATED
MxPhong& MxPhong::setLightColors(const Containers::ArrayView<const Magnum::Color4> colors) {
    Containers::Array<Magnum::Color3> threeComponent{Containers::NoInit, colors.size()};
    for(std::size_t i = 0; i != colors.size(); ++i)
        threeComponent[i] = colors[i].rgb();
    setLightColors(threeComponent);
    return *this;
}

MxPhong& MxPhong::setLightColors(const std::initializer_list<Magnum::Color4> colors) {
    CORRADE_IGNORE_DEPRECATED_PUSH
    return setLightColors(Containers::arrayView(colors));
    CORRADE_IGNORE_DEPRECATED_POP
}
#endif

MxPhong& MxPhong::setLightColors(const std::initializer_list<Magnum::Color3> colors) {
    return setLightColors(Containers::arrayView(colors));
}

MxPhong& MxPhong::setLightColor(const UnsignedInt id, const Magnum::Color3& color) {
    CORRADE_ASSERT(id < _lightCount,
        "Shaders::MxPhong::setLightColor(): light ID" << id << "is out of bounds for" << _lightCount << "lights", *this);
    setUniform(_lightColorsUniform + id, color);
    return *this;
}

#ifdef MAGNUM_BUILD_DEPRECATED
MxPhong& MxPhong::setLightColor(UnsignedInt id, const Magnum::Color4& color) {
    return setLightColor(id, color.rgb());
}

MxPhong& MxPhong::setLightColor(const Magnum::Color4& color) {
    /* Use the list variant to check the shader really has just one light */
    return setLightColors({color.rgb()});
}
#endif

MxPhong& MxPhong::setLightSpecularColors(const Containers::ArrayView<const Magnum::Color3> colors) {
    CORRADE_ASSERT(_lightCount == colors.size(),
        "Shaders::MxPhong::setLightSpecularColors(): expected" << _lightCount << "items but got" << colors.size(), *this);
    if(_lightCount) setUniform(_lightSpecularColorsUniform, colors);
    return *this;
}

MxPhong& MxPhong::setLightSpecularColors(const std::initializer_list<Magnum::Color3> colors) {
    return setLightSpecularColors(Containers::arrayView(colors));
}

MxPhong& MxPhong::setLightSpecularColor(const UnsignedInt id, const Magnum::Color3& color) {
    CORRADE_ASSERT(id < _lightCount,
        "Shaders::MxPhong::setLightSpecularColor(): light ID" << id << "is out of bounds for" << _lightCount << "lights", *this);
    setUniform(_lightSpecularColorsUniform + id, color);
    return *this;
}

MxPhong& MxPhong::setLightRanges(const Containers::ArrayView<const Float> ranges) {
    CORRADE_ASSERT(_lightCount == ranges.size(),
        "Shaders::MxPhong::setLightRanges(): expected" << _lightCount << "items but got" << ranges.size(), *this);
    if(_lightCount) setUniform(_lightRangesUniform, ranges);
    return *this;
}

MxPhong& MxPhong::setLightRanges(const std::initializer_list<Float> ranges) {
    return setLightRanges(Containers::arrayView(ranges));
}

MxPhong& MxPhong::setLightRange(const UnsignedInt id, const Float range) {
    CORRADE_ASSERT(id < _lightCount,
        "Shaders::MxPhong::setLightRange(): light ID" << id << "is out of bounds for" << _lightCount << "lights", *this);
    setUniform(_lightRangesUniform + id, range);
    return *this;
}

Debug& operator<<(Debug& debug, const MxPhong::Flag value) {
    debug << "Shaders::MxPhong::Flag" << Debug::nospace;

    switch(value) {
        /* LCOV_EXCL_START */
        #define _c(v) case MxPhong::Flag::v: return debug << "::" #v;
        _c(AmbientTexture)
        _c(DiffuseTexture)
        _c(SpecularTexture)
        _c(NormalTexture)
        _c(Bitangent)
        _c(AlphaMask)
        _c(VertexColor)
        _c(TextureTransformation)
        #ifndef MAGNUM_TARGET_GLES2
        _c(InstancedObjectId)
        _c(ObjectId)
        #endif
        _c(InstancedTransformation)
        _c(InstancedTextureOffset)
        #undef _c
        /* LCOV_EXCL_STOP */
    }

    return debug << "(" << Debug::nospace << reinterpret_cast<void*>(UnsignedByte(value)) << Debug::nospace << ")";
}

Debug& operator<<(Debug& debug, const MxPhong::Flags value) {
    return Containers::enumSetDebugOutput(debug, value, "Shaders::MxPhong::Flags{}", {
        MxPhong::Flag::AmbientTexture,
        MxPhong::Flag::DiffuseTexture,
        MxPhong::Flag::SpecularTexture,
        MxPhong::Flag::NormalTexture,
        MxPhong::Flag::Bitangent,
        MxPhong::Flag::AlphaMask,
        MxPhong::Flag::VertexColor,
        MxPhong::Flag::InstancedTextureOffset, /* Superset of TextureTransformation */
        MxPhong::Flag::TextureTransformation,
        #ifndef MAGNUM_TARGET_GLES2
        MxPhong::Flag::InstancedObjectId, /* Superset of ObjectId */
        MxPhong::Flag::ObjectId,
        #endif
        MxPhong::Flag::InstancedTransformation});
}

}}
