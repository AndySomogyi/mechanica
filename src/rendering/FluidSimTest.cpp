/*******************************************************************************
    This file is part of Mechanica.

    Based on Magnum and mdcore examples

    Original authors — credit is appreciated but not required:
    2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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

#include "FluidSimTest.h"

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
#include <MxSimulator.h>
#include <MxUniverse.h>



// include some standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <time.h>
#include <random>
#include <iostream>
#include "cycle.h"
#include "mdcore_single.h"

/* MPI headers. */
#ifdef HAVE_MPI
#include <mpi.h>
#endif

/* FFTW3 headers. */
#ifdef HAVE_FFTW3
#include <complex.h>
#include <fftw3.h>
#endif





using namespace Magnum;
using namespace Math::Literals;

namespace {
    constexpr Float ParticleRadius = 0.2f;
}

static const double BOUNDARY_SCALE = 1.05;

static std::vector<Vector3> createCubicLattice(float length, float spacing) {

    if(spacing > length) {
        return {Vector3{(float)(length / 2.), (float)(length / 2.), (float)(length / 2.)}};
    }

    std::vector<Vector3> result;
    int n = ceil(length / spacing);
    float s = length / (n - 1);
    //float l2 = length / 2;

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                float x = i * s;
                float y = j * s;
                float z = k * s;
                //std::cout << "{" << x << ", " << y << ", " << z << "}\n";
                result.push_back(Vector3{x, y, z});
            }
        }
    }

    return result;
}





FluidSimTest::FluidSimTest(const Arguments& arguments) :
        Magnum::Platform::GlfwApplication{arguments, NoCreate},
        mxWindow{NULL} {

    for(int i = 0; i < arguments.argc; ++i) {
        std::cout << "arg[" << i << "]: " << arguments.argv[i] << std::endl;
        std::string arg(arguments.argv[i]);

        if(strcmp("-nw", arguments.argv[i]) == 0) {
            display = false;
        }

        else if(arg.find("-parts=") == 0) {
            try {
                nParticles = std::stoi(arg.substr(7));
                std::cout << "particles: " << nParticles << std::endl;
            }
            catch (...) {
                std::cerr << "ERROR invalid particles!\n";
            }
        }

        else if(arg.find("-dt=") == 0) {
            try {
                dt = std::stof(arg.substr(4));
                std::cout << "dt: " << dt << std::endl;
            }
            catch (...) {
                std::cerr << "ERROR invalid dt!\n";
            }
        }

        else if(arg.find("-temp=") == 0) {
            try {
                temp = std::stof(arg.substr(6));
                std::cout << "temp: " << temp << std::endl;
            }
            catch (...) {
                std::cerr << "ERROR invalid temp!\n";
            }
        }
    }
        
    /* Initialize scene particles */
        MxUniverseConfig conf;
        conf.origin = origin;
        conf.dim = dim;
        conf.nParticles = nParticles;
        conf.dt = 0.01;
        conf.temp = 0.01;
        
        universe_init(conf);
        
        example_argon(conf);
    
    if(display) {
        /* Setup window */
        const Vector2 dpiScaling = this->dpiScaling({});
        Configuration conf;
        conf.setTitle("SPH Testing")
            .setSize(conf.size(), dpiScaling)
            .setWindowFlags(Configuration::WindowFlag::Resizable);
        GLConfiguration glConf;
        glConf.setSampleCount(dpiScaling.max() < 2.0f ? 8 : 2);
        if(!tryCreate(conf, glConf)) {
            create(conf, glConf.setSampleCount(0));
        }

        // TODO temporary hack to initialized mechanica becasue we create
        // context here.
        Mx_Initialize(0);
        
        const auto viewportSize = GL::defaultFramebuffer.viewport().size();

        const auto viewportSize2  = GL::defaultFramebuffer.viewport().size();
        mxWindow = new MxGlfwWindow(window());
        

        /* Drawable particles */
        _drawableParticles.reset(new MxUniverseRenderer{mxWindow});
    }
    


    /* Reset domain */
    if(_dynamicBoundary) _boundaryOffset = 0.0f;


    if(display) {
        /* Trigger drawable object to upload particles to the GPU */
        _drawableParticles->setDirty();
    }

    /* Start the timer */
    _timeline.start();
}

void FluidSimTest::drawEvent() {
    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);



    /* Pause simulation if the mouse was pressed (camera is moving around).
       This avoid freezing GUI while running the simulation */
    if(!_pausedSimulation && !_mousePressed) {
        /* Adjust the substep number to maximize CPU usage each frame */
        const Float lastAvgStepTime = _timeline.previousFrameDuration()/Float(_substeps);
        const Int newSubsteps = lastAvgStepTime > 0 ? Int(1.0f/60.0f/lastAvgStepTime) + 1 : 1;
        if(Math::abs(newSubsteps - _substeps) > 1) _substeps = newSubsteps;

        for(Int i = 0; i < _substeps; ++i) simulationStep();
    }


        /* Draw particles */
        _drawableParticles->draw();





    swapBuffers();
    _timeline.nextFrame();

    /* Run next frame immediately */
    redraw();
}















void FluidSimTest::initializeScene() {
    


}

int FluidSimTest::nonDisplayExec()
{
    for(currentStep = 0; currentStep < nSteps; ++currentStep) {
        simulationStep();
    }
    return 0;
}

int FluidSimTest::exec()
{
    if(display) {
        return GlfwApplication::exec();
    }
    else {
        return nonDisplayExec();
    }
}

void FluidSimTest::simulationStep() {
    static Float offset = 0.0f;
    if(_dynamicBoundary) {
        /* Change fluid boundary */
        static Float step = 2.0e-3f;
        if(_boundaryOffset > 1.0f || _boundaryOffset < 0.0f) {
            step *= -1.0f;
        }
        _boundaryOffset += step;
        offset = Math::lerp(0.0f, 0.5f, Animation::Easing::quadraticInOut(_boundaryOffset));
    }


    MxUniverse_Step(0,0);
    
    currentStep += 1;
}

void FluidSimTest::viewportEvent(ViewportEvent& event) {
    /* Resize the main framebuffer */
    //GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});



    //event.
    //( int x, int y);

    _drawableParticles->viewportEvent(event.framebufferSize().x(), event.framebufferSize().y());


}

FluidSimTest::~FluidSimTest() {
    std::cout << MX_FUNCTION << std::endl;
}











