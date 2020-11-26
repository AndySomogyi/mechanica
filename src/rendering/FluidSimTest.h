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

#include <Magnum/Animation/Easing.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Version.h>
#include <Magnum/SceneGraph/Scene.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/Drawable.h>
#include <Magnum/Math/FunctionsBatch.h>

#include <iostream>

#include <string>

#include <Mechanica.h>
#include <rendering/WireframeObjects.h>

#include <Corrade/Containers/Pointer.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>
#include <Magnum/Timeline.h>
#include <rendering/MxUniverseRenderer.h>
#include <rendering/MxGlfwWindow.h>

using namespace Magnum;



using Object3D = SceneGraph::Object<SceneGraph::MatrixTransformation3D>;
using Scene3D  = SceneGraph::Scene<SceneGraph::MatrixTransformation3D>;

class WireframeGrid;
class WireframeBox;
class MxUniverseRendererOld;

#include <MxUniverse.h>



class CAPI_EXPORT FluidSimTest: public Platform::GlfwApplication {
    public:
        explicit FluidSimTest(const Arguments& arguments);

        int nonDisplayExec();

        int exec();

        virtual ~FluidSimTest();


    protected:

        void drawEvent() override;



        /* Fluid simulation helper functions */
        void showMenu() {};
        void initializeScene();
        void simulationStep();

        /* Window control */
        bool _showMenu = true;
        //ImGuiIntegration::Context _imGuiContext{NoCreate};





        Int _substeps = 1;
        bool _pausedSimulation = false;
        bool _mousePressed = false;
        bool _dynamicBoundary = true;
        Float _boundaryOffset = 0.0f; /* For boundary animation */

        /* Drawable particles */
        Containers::Pointer<MxUniverseRenderer> _drawableParticles;



        /* Timeline to adjust number of simulation steps per frame */
        Timeline _timeline;


        float sideLength = 10.0;

        bool display = true;

        int nSteps = 10;
        int currentStep = 0;

        int nParticles = 1000;



        float dt = 0.01;
        float temp = 1;

        Vector3 origin = {0.0, 0.0, 0.0};
        Vector3 dim = {10., 10., 10.};

        MxGlfwWindow *mxWindow;

        void viewportEvent(ViewportEvent& event) override;
};

