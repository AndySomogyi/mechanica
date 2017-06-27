#ifndef Magnum_Trade_AssimpImporter_h
#define Magnum_Trade_AssimpImporter_h
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

/** @file
 * @brief Class @ref Magnum::Trade::AssimpImporter
 */

#include <Magnum/Trade/AbstractImporter.h>

#include "visibility.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>

#include <unordered_map>

namespace Magnum { namespace Trade {

/**
@brief Assimp importer

Imports various formats using [Assimp](http://assimp.org).

Supports importing of scene, object, camera, mesh, texture and image data.

It is built if `WITH_ASSIMPIMPORTER` is enabled when building Magnum Plugins.
To use dynamic plugin, you need to load `AssimpImporter` plugin from
`MAGNUM_PLUGINS_IMPORTER_DIR`. To use static plugin, you need to request
`AssimpImporter` component of `MagnumPlugins` package in CMake and link to
`MagnumPlugins::AssimpImporter`. See @ref building-plugins, @ref cmake-plugins
and @ref plugins for more information.

### Behaviour and limitations

#### Material import

- Only materials with shading mode `aiShadingMode_Phong` are supported
- Only the first diffuse/specular/ambient texture is loaded


*/
class MAGNUM_TRADE_ASSIMPIMPORTER_EXPORT AssimpImporter: public AbstractImporter {
    public:
        /**
         * @brief Default constructor
         *
         * In case you want to open images, use
         * @ref AssimpImporter(PluginManager::Manager<AbstractImporter>&)
         * instead.
         */
        explicit AssimpImporter();

        /**
         * @brief Constructor
         *
         * The plugin needs access to plugin manager for importing images.
         */
        explicit AssimpImporter(PluginManager::Manager<AbstractImporter>& manager);

        /** @brief Plugin manager constructor */
        explicit AssimpImporter(PluginManager::AbstractManager& manager, const std::string& plugin);

        ~AssimpImporter();

    private:
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL Features doFeatures() const override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL bool doIsOpened() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL void doOpenData(Containers::ArrayView<const char> data) override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL void doOpenFile(const std::string& filename) override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL void doClose() override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL Int doDefaultScene() override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doSceneCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::optional<SceneData> doScene(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doCameraCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::optional<CameraData> doCamera(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doObject3DCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL Int doObject3DForName(const std::string& name) override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::string doObject3DName(UnsignedInt id) override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::unique_ptr<ObjectData3D> doObject3D(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doLightCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::optional<LightData> doLight(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doMesh3DCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::optional<MeshData3D> doMesh3D(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doMaterialCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL Int doMaterialForName(const std::string& name) override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::string doMaterialName(UnsignedInt id) override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::unique_ptr<AbstractMaterialData> doMaterial(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doTextureCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::optional<TextureData> doTexture(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL UnsignedInt doImage2DCount() const override;
        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL std::optional<ImageData2D> doImage2D(UnsignedInt id) override;

        MAGNUM_TRADE_ASSIMPIMPORTER_LOCAL const void* doImporterState() const override;

        void createNodeIndices();

        class Assimp::Importer _importer;
        const aiScene* _scene = nullptr;
        std::vector<aiNode*> _nodes;
        std::unordered_map<aiNode*, UnsignedInt> _nodeIndices;
};

}}

#endif
