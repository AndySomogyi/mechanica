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

#include <Corrade/Containers/Pointer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/Timeline.h>

using namespace Magnum;



using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D  = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

class SPHSolver;
class WireframeGrid;
class WireframeBox;
class MxUniverseRenderer;

#include <MxUniverse.h>



class FluidSimApp: public Platform::GlfwApplication {
    public:
        explicit FluidSimApp(const Arguments& arguments);

        int nonDisplayExec();

        int exec();

    protected:
        void viewportEvent(ViewportEvent& event) override;
        void keyPressEvent(KeyEvent& event) override;
        void keyReleaseEvent(KeyEvent& event) override;
        void mousePressEvent(MouseEvent& event) override;
        void mouseReleaseEvent(MouseEvent& event) override;
        void mouseMoveEvent(MouseMoveEvent& event) override;
        void mouseScrollEvent(MouseScrollEvent& event) override;
        void textInputEvent(TextInputEvent& event) override;
        void drawEvent() override;

        /* Helper functions for camera movement */
        Float depthAt(const Vector2i& windowPosition);
        Vector3 unproject(const Vector2i& windowPosition, Float depth) const;

        /* Fluid simulation helper functions */
        void showMenu() {};
        void initializeScene();
        void simulationStep();

        /* Window control */
        bool _showMenu = true;
        //ImGuiIntegration::Context _imGuiContext{NoCreate};

        /* Scene and drawable group must be constructed before camera and other
        scene objects */
        Containers::Pointer<Scene3D> _scene;
        Containers::Pointer<SceneGraph::DrawableGroup3D> _drawableGroup;

        /* Camera helpers */
        Vector3 _defaultCamPosition{0.0f, 1.5f, 8.0f};
        Vector3 _defaultCamTarget{0.0f, 0.0f, 0.0f};

        Vector2i _prevMousePosition;
        Vector3  _rotationPoint, _translationPoint;
        Float _lastDepth;
        Containers::Pointer<Object3D> _objCamera;
        Containers::Pointer<SceneGraph::Camera3D> _camera;

        /* Fluid simulation system */
        Containers::Pointer<SPHSolver> _fluidSolver;
        Containers::Pointer<WireframeBox> _drawableBox;
        Int _substeps = 1;
        bool _pausedSimulation = false;
        bool _mousePressed = false;
        bool _dynamicBoundary = true;
        Float _boundaryOffset = 0.0f; /* For boundary animation */

        /* Drawable particles */
        Containers::Pointer<MxUniverseRenderer> _drawableParticles;

        /* Ground grid */
        Containers::Pointer<WireframeGrid> _grid;

        /* Timeline to adjust number of simulation steps per frame */
        Timeline _timeline;


        float sideLength = 1.0;

        bool display = true;

        int nSteps = 10;
        int currentStep = 0;
};

