/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017
              Vladimír Vondruš <mosra@centrum.cz>
    Copyright © 2017 Jonathan Hale <squareys@googlemail.com>

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

#include "AssimpImporter.h"

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <Corrade/Containers/ArrayView.h>
#include <Corrade/Utility/Directory.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/Math/Vector.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/PixelStorage.h>
#include <Magnum/Trade/CameraData.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/LightData.h>
#include <Magnum/Trade/MeshData3D.h>
#include <Magnum/Trade/MeshObjectData3D.h>
#include <Magnum/Trade/PhongMaterialData.h>
#include <Magnum/Trade/SceneData.h>
#include <Magnum/Trade/TextureData.h>

#include <assimp/postprocess.h>
#include <assimp/material.h>

namespace Magnum { namespace Math { namespace Implementation {

template<> struct VectorConverter<3, Float, aiColor3D> {
    static Vector<3, Float> from(const aiColor3D& other) {
        return {other.r, other.g, other.b};
    }

    static aiColor3D to(const Vector<3, Float>& other) {
        return {other[0], other[1], other[2]};
    }
};

}}}

namespace Magnum { namespace Trade {


using namespace Magnum::Math::Literals;

AssimpImporter::AssimpImporter() = default;

AssimpImporter::AssimpImporter(PluginManager::Manager<AbstractImporter>& manager): AbstractImporter(manager) {}

AssimpImporter::AssimpImporter(PluginManager::AbstractManager& manager, const std::string& plugin): AbstractImporter(manager, plugin) {}

AssimpImporter::~AssimpImporter() = default;

auto AssimpImporter::doFeatures() const -> Features { return Feature::OpenData; }

bool AssimpImporter::doIsOpened() const { return !!_scene; }

void AssimpImporter::doOpenData(const Containers::ArrayView<const char> data) {
    _scene = _importer.ReadFileFromMemory(data.data(), data.size(), aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    createNodeIndices();
}

void AssimpImporter::doOpenFile(const std::string& filename) {
    _scene = _importer.ReadFile(filename, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    createNodeIndices();
}

void AssimpImporter::doClose() { delete _scene; _scene = nullptr; _nodes.clear(); }

Int AssimpImporter::doDefaultScene() { return 0; }

UnsignedInt AssimpImporter::doSceneCount() const { return 1; }

std::optional<SceneData> AssimpImporter::doScene(UnsignedInt) {
    const aiNode* root = _scene->mRootNode;
    return SceneData{{}, {0}, root};
}

UnsignedInt AssimpImporter::doCameraCount() const {
    return _scene->mNumCameras;
}

std::optional<CameraData> AssimpImporter::doCamera(UnsignedInt id) {
    const aiCamera* cam = _scene->mCameras[id];
    /** @todo aspect and up vector are not used... */
    return CameraData(Rad(cam->mHorizontalFOV), cam->mClipPlaneNear, cam->mClipPlaneFar, cam);
}

UnsignedInt AssimpImporter::doObject3DCount() const {
    return _nodes.size();
}

Int AssimpImporter::doObject3DForName(const std::string& name) {
    const auto found = _scene->mRootNode->FindNode(aiString(name));
    return found ? -1 : 42; //static_cast<Long>(found); TODO
}

std::string AssimpImporter::doObject3DName(const UnsignedInt id) {
    return _nodes[id]->mName.C_Str();
}

std::unique_ptr<ObjectData3D> AssimpImporter::doObject3D(const UnsignedInt id) {
    /** @todo support for bone nodes */
    const aiNode* node = _nodes[id];

    /** Gather child indices */
    std::vector<UnsignedInt> children;
    children.reserve(node->mNumChildren);
    for(auto child : Containers::ArrayView<aiNode*>(node->mChildren, node->mNumChildren)) {
        children.push_back(_nodeIndices[child]);
    }

    const Matrix4 transformation = Matrix4::from(reinterpret_cast<const float*>(&node->mTransformation));
    return std::unique_ptr<ObjectData3D>{new ObjectData3D(children, transformation, node)};
}

UnsignedInt AssimpImporter::doLightCount() const {
    return _scene->mNumLights;
}

std::optional<LightData> AssimpImporter::doLight(UnsignedInt id) {
    const aiLight* l = _scene->mLights[id];

    LightData::Type lightType;
    if(l->mType == aiLightSource_DIRECTIONAL) {
        lightType = LightData::Type::Infinite;
    } else if(l->mType == aiLightSource_POINT) {
        lightType = LightData::Type::Point;
    } else if(l->mType == aiLightSource_SPOT) {
        lightType = LightData::Type::Spot;
    } else {
        Error() << "Undefined light type is not supported.";
        return {};
    }


    Color3 ambientColor = Color3::from(reinterpret_cast<const float*>(&l->mColorAmbient));

    /** @todo angle inner/outer cone, linear/quadratic/constant atteniution, diffuse/specular color are not used */
    return LightData(lightType, ambientColor, 1.0f, l);
}

UnsignedInt AssimpImporter::doMesh3DCount() const {
    return _scene->mNumMeshes;
}

std::optional<MeshData3D> AssimpImporter::doMesh3D(const UnsignedInt id) {
    const aiMesh* mesh = _scene->mMeshes[id];

    MeshPrimitive primitive;
    if(mesh->mPrimitiveTypes == aiPrimitiveType_POINT) {
        primitive = MeshPrimitive::Points;
    } else if(mesh->mPrimitiveTypes == aiPrimitiveType_LINE) {
        primitive = MeshPrimitive::Lines;
    } else if(mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE) {
        primitive = MeshPrimitive::Triangles;
    }

    std::vector<std::vector<Vector3>> positions;
    std::vector<std::vector<Vector2>> textureCoordinates;
    std::vector<std::vector<Vector3>> normals;
    std::vector<std::vector<Color4>> colors;

    positions.emplace_back(); // FIXME(C++17) assign to pos directly
    auto pos = positions[0];
    pos.reserve(mesh->mNumVertices);
    pos.insert(pos.end(), reinterpret_cast<Vector3*>(mesh->mVertices),
            reinterpret_cast<Vector3*>(mesh->mVertices) + mesh->mNumVertices*sizeof(Vector3));

    normals.emplace_back(); // FIXME(C++17) assign to pos directly
    auto norms = normals[0];
    norms.reserve(mesh->mNumVertices);
    norms.insert(norms.end(), reinterpret_cast<Vector3*>(mesh->mNormals),
            reinterpret_cast<Vector3*>(mesh->mNormals) + mesh->mNumVertices*sizeof(Vector3));

    /** @todo only first uv layer (or "channel") supported) */
    textureCoordinates.reserve(mesh->GetNumUVChannels());
    for (size_t layer = 0; layer < mesh->GetNumUVChannels(); ++layer) {
        textureCoordinates.emplace_back();
        auto texCoords = textureCoordinates[layer];
        texCoords.reserve(mesh->mNumVertices);
        for(size_t i; i < mesh->mNumVertices; ++i) {
            texCoords.emplace_back(mesh->mTextureCoords[layer][i].x, mesh->mTextureCoords[layer][i].y);
        }
    }

    colors.reserve(mesh->GetNumColorChannels());
    for (size_t layer = 0; layer < mesh->GetNumColorChannels(); ++layer) {
        colors.emplace_back();
        auto cols = colors[layer];
        cols.reserve(mesh->mNumVertices);
        cols.insert(cols.end(), reinterpret_cast<Color4*>(mesh->mColors[layer]),
                reinterpret_cast<Color4*>(mesh->mColors[layer]) + mesh->mNumVertices*sizeof(Color4));
    }

    /* Import indices */
    std::vector<UnsignedInt> indices;
    indices.reserve(mesh->mNumFaces*3);

    for(size_t faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
        const aiFace& face = mesh->mFaces[faceIndex];

        CORRADE_ASSERT(face.mNumIndices == 3, "Expected number of indices per face to always be 3, since using triangulation post processing of assimp", {});
        for(const int i: {0, 1, 2}) {
            indices.emplace_back(face.mIndices[i]);
        }
    }

    return MeshData3D(primitive, std::move(indices), std::move(positions), std::move(normals), std::move(textureCoordinates), std::move(colors), mesh);
}

UnsignedInt AssimpImporter::doMaterialCount() const { return _scene->mNumMaterials; }

Int AssimpImporter::doMaterialForName(const std::string& name) {
    const aiString assimpName{name}; /* wrap in aiString for == */
    aiString matName;

    // TODO: Optionally, create a hashmap when opening the file, or create it lazily
    for(size_t i = 0; i < _scene->mNumMaterials; ++i) {
        if(_scene->mMaterials[i]->Get(AI_MATKEY_NAME, matName) == AI_SUCCESS) {
            if(matName == assimpName) {
                return i;
            }
        }
    }

    return -1;
}

std::string AssimpImporter::doMaterialName(const UnsignedInt id) {
    const aiMaterial* mat = _scene->mMaterials[id];
    aiString name;
    mat->Get(AI_MATKEY_NAME, name);

    return name.C_Str();
}

std::unique_ptr<AbstractMaterialData> AssimpImporter::doMaterial(const UnsignedInt id) {
    /* Put things together */
    const aiMaterial* mat = _scene->mMaterials[id];

    aiShadingMode shadingMode;

    if(mat->Get(AI_MATKEY_SHADING_MODEL, shadingMode) == AI_SUCCESS) {
        if(shadingMode != aiShadingMode_Phong) {
            Error() << "Unsupported shading mode";
            return {};
        }
    }

    PhongMaterialData::Flags flags;
    Float shininess;
    aiString texturePath;
    aiColor3D color;

    if(mat->Get(AI_MATKEY_TEXTURE(aiTextureType_AMBIENT, 0), texturePath) == AI_SUCCESS) {
        flags |= PhongMaterialData::Flag::AmbientTexture;
    }
    if(mat->Get(AI_MATKEY_TEXTURE(aiTextureType_DIFFUSE, 0), texturePath) == AI_SUCCESS) {
        flags |= PhongMaterialData::Flag::DiffuseTexture;
    }
    if(mat->Get(AI_MATKEY_TEXTURE(aiTextureType_SPECULAR, 0), texturePath) == AI_SUCCESS) {
        flags |= PhongMaterialData::Flag::SpecularTexture;
    }
    /* @todo many more types supported in assimp */

    mat->Get(AI_MATKEY_SHININESS, shininess); /* Key always present, default 0.0f */

    std::unique_ptr<PhongMaterialData> data{new PhongMaterialData(flags, shininess, mat)};

    mat->Get(AI_MATKEY_COLOR_AMBIENT, color); /* Key always present, default black */
    if(flags & PhongMaterialData::Flag::AmbientTexture)
        data->ambientColor() = Color3(color);

    mat->Get(AI_MATKEY_COLOR_DIFFUSE, color); /* Key always present, default black */
    if(flags & PhongMaterialData::Flag::DiffuseTexture)
        data->diffuseColor() = Color3(color);

    mat->Get(AI_MATKEY_COLOR_SPECULAR, color); /* Key always present, default black */
    if(flags & PhongMaterialData::Flag::SpecularTexture)
        data->specularColor() = Color3(color);

    return std::move(data);
}

UnsignedInt AssimpImporter::doTextureCount() const { return _scene->mNumTextures; }

std::optional<TextureData> AssimpImporter::doTexture(const UnsignedInt id) {
    const aiTexture* texture = _scene->mTextures[id];

    // TODO: Get texture parameters from Material

    return TextureData{TextureData::Type::Texture2D, Sampler::Filter::Linear, Sampler::Filter::Linear, {Sampler::Mipmap::Linear}, Sampler::Wrapping::ClampToEdge, id, texture};
}

UnsignedInt AssimpImporter::doImage2DCount() const { return _scene->mNumTextures; }

std::optional<ImageData2D> AssimpImporter::doImage2D(const UnsignedInt id) {
    const aiTexture* texture = _scene->mTextures[id];

    if(texture->mHeight == 0) {
        /* Compressed image data */
        std::string importerName;
        if(texture->CheckFormat("dds")) {
            importerName = "DdsImporter";
        } else if(texture->CheckFormat("jpg")) {
        } else if(texture->CheckFormat("pcx")) {
        } else if(texture->CheckFormat("png")) {
        }
        return {};
    } else {
        /* Uncompressed image data */
        const Vector2i dimensions{Int(texture->mHeight), Int(texture->mWidth)};
        size_t size = dimensions.product()*4;
        Containers::Array<char> data{Containers::NoInit, size};
        std::memcpy(texture->pcData, data.data(), size);

        return ImageData2D(PixelFormat::RGBA, PixelType::UnsignedByte, dimensions, std::move(data), texture);
    }
}

const void* AssimpImporter::doImporterState() const {
    return _scene;
}

void AssimpImporter::createNodeIndices() {
    if(!_scene) {
        /* Scene was not imported correctly by assimp */
        return;
    }
    if(!_nodes.empty()) {
        /* Already done previously */
        return;
    }

    /* reserve memory for the amount of nodes we know will at least be present */
    aiNode* root = _scene->mRootNode;
    if(!root) {
        return; /* Nothing to do */
    }

    _nodes.reserve(root->mNumChildren+1); /* Children + root itself */
    _nodes.push_back(root);

    _nodeIndices.reserve(root->mNumChildren+1);

    /* insert may invalidate iterators, so we use indices here. */
    for(size_t i = 0; i < _nodes.size(); ++i) {
        aiNode* node = _nodes[i];
        _nodeIndices[node] = UnsignedInt(i);

        Containers::ArrayView<aiNode*> children(node->mChildren, node->mNumChildren);
        _nodes.insert(_nodes.end(), children.begin(), children.end());

    }
}

}}
