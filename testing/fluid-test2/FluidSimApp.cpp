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
#include <rendering/MxUniverseRendererOld.h>
#include <rendering/WireframeObjects.h>

#include "FluidSimApp.h"


using namespace Magnum;
using namespace Math::Literals;

namespace {
constexpr Float ParticleRadius = 0.2f;
}

static void engineStep();

static int initArgon (const Vector3 &origin, const Vector3 &dim,
        int nParticles, double dt = 0.005, float temp = 100 );

static const double BOUNDARY_SCALE = 1.05;

static std::vector<Vector3> createCubicLattice(float length, float spacing) {

    if(spacing > length) {
        return {Vector3{length / 2.,length / 2., length / 2.}};
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

#include <random>
#include <iostream>


static std::vector<Vector3> fillCubeRandom(const Vector3 &corner1, const Vector3 &corner2, int nParticles) {
    std::vector<Vector3> result;

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> disx(corner1[0], corner2[0]);
    std::uniform_real_distribution<> disy(corner1[1], corner2[1]);
    std::uniform_real_distribution<> disz(corner1[2], corner2[2]);

    for(int i = 0; i < nParticles; ++i) {
        result.push_back(Vector3{disx(gen), disy(gen), disz(gen)});

    }

    return result;
}

FluidSimApp::FluidSimApp(const Arguments& arguments): Platform::Application{arguments, NoCreate} {

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


        GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
        GL::Renderer::setClearColor(Color3{0.35f});

        /* Loop at 60 Hz max */
        setSwapInterval(1);
        //setMinimalLoopPeriod(16);


        center = (dim + origin) / 2.;


        /* Setup scene objects and camera */

        /* Setup scene objects */
        _scene.reset(new Scene3D{});
        _drawableGroup.reset(new SceneGraph::DrawableGroup3D{});

        /* Configure camera */
        _objCamera.reset(new Object3D{ _scene.get() });

        const auto viewportSize = GL::defaultFramebuffer.viewport().size();
        _camera.reset(new SceneGraph::Camera3D{ *_objCamera });

        _camera->setProjectionMatrix(Matrix4::perspectiveProjection(45.0_degf, Vector2{ viewportSize }.aspectRatio(), 0.01f, 1000.0f))
                    .setViewport(viewportSize);

        /* Set default camera parameters */
        _defaultCamPosition = Vector3(2*sideLength, 2*sideLength, 3 * sideLength);

        _defaultCamTarget   = {0,0,0};

        _objCamera->setTransformation(Matrix4::lookAt(_defaultCamPosition, _defaultCamTarget, Vector3(0, 1, 0)));

        /* Initialize depth to the value at scene center */
        _lastDepth = ((_camera->projectionMatrix() * _camera->cameraMatrix()).transformPoint({}).z() + 1.0f) * 0.5f;


        /* Setup ground grid */

        // grid is ???
        _grid.reset(new WireframeGrid(_scene.get(), _drawableGroup.get()));
        _grid->transform(Matrix4::scaling(Vector3(1.f))  );


        /* Setup fluid solver */



        /* Simulation domain box */
        /* Transform the box to cover the region [0, 0, 0] to [3, 3, 1] */
        _drawableBox.reset(new WireframeBox(_scene.get(), _drawableGroup.get()));

        // box is cube of side length 2 centered at origin
        _drawableBox->transform(Matrix4::scaling(Vector3(sideLength / 2)) );
        _drawableBox->setColor(Color3(1, 1, 0));

        /* Drawable particles */
        _drawableParticles.reset(new MxUniverseRendererOld{ParticleRadius});

        _drawableParticles->setModelViewTransform(Matrix4::translation(-center));
    }

    /* Initialize scene particles */
    initArgon(origin, dim, nParticles, 0.01, 0.01);

    /* Reset domain */
    if(_dynamicBoundary) _boundaryOffset = 0.0f;


    if(display) {
        /* Trigger drawable object to upload particles to the GPU */
        _drawableParticles->setDirty();
    }

    /* Start the timer */
    _timeline.start();
}

void FluidSimApp::drawEvent() {
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

    std::cout << "cells: " << engine_get()->s.nr_cells << ", parts: " <<  engine_get()->s.nr_parts << std::endl;

    /* Draw objects */
    {
        /* Trigger drawable object to update the particles to the GPU */
        _drawableParticles->setDirty();
        /* Draw particles */
        _drawableParticles->draw(_camera, framebufferSize());

        /* Draw other objects (ground grid) */
        _camera->draw(*_drawableGroup);
    }

    /* Menu for parameters */
    if(_showMenu) showMenu();



    swapBuffers();
    _timeline.nextFrame();

    /* Run next frame immediately */
    redraw();
}

void FluidSimApp::viewportEvent(ViewportEvent& event) {
    /* Resize the main framebuffer */
    GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});

    /* Relayout ImGui */
    //_imGuiContext.relayout(Vector2{event.windowSize()}/event.dpiScaling(), event.windowSize(), event.framebufferSize());

    /* Recompute the camera's projection matrix */
    _camera->setViewport(event.framebufferSize());
}

void FluidSimApp::keyPressEvent(Platform::GlfwApplication::KeyEvent& event) {
    switch(event.key()) {
    case KeyEvent::Key::H:
        _showMenu ^= true;
        event.setAccepted(true);
        break;
    case KeyEvent::Key::R:
        initializeScene();
        event.setAccepted(true);
        break;
    case KeyEvent::Key::Space:
        _pausedSimulation ^= true;
        event.setAccepted(true);
        break;
    default:
        break;
        //            if(_imGuiContext.handleKeyPressEvent(event)) {
        //                event.setAccepted(true);
        //            }
    }
}

void FluidSimApp::keyReleaseEvent(KeyEvent& event) {
    //    if(_imGuiContext.handleKeyReleaseEvent(event)) {
    //        event.setAccepted(true);
    //        return;
    //    }
}

void FluidSimApp::mousePressEvent(MouseEvent& event) {


    if((event.button() != MouseEvent::Button::Left)
            && (event.button() != MouseEvent::Button::Right)) {
        return;
    }

    /* Update camera */
    {
        _prevMousePosition = event.position();
        const Float currentDepth = depthAt(event.position());
        const Float depth = currentDepth == 1.0f ? _lastDepth : currentDepth;
        _translationPoint = unproject(event.position(), depth);

        /* Update the rotation point only if we're not zooming against infinite
           depth or if the original rotation point is not yet initialized */
        if(currentDepth != 1.0f || _rotationPoint.isZero()) {
            _rotationPoint = _translationPoint;
            _lastDepth = depth;
        }
    }

    _mousePressed = true;
}

void FluidSimApp::mouseReleaseEvent(MouseEvent& event) {
    _mousePressed = false;

}

void FluidSimApp::mouseMoveEvent(MouseMoveEvent& event) {

    if(!(event.buttons() & MouseMoveEvent::Button::Left)
            && !(event.buttons() & MouseMoveEvent::Button::Right)) {
        return;
    }

    const Vector2 delta = 3.0f*Vector2{event.position() - _prevMousePosition}/Vector2{framebufferSize()};
    _prevMousePosition = event.position();

    if(event.buttons() & MouseMoveEvent::Button::Left) {
        _objCamera->transformLocal(
                Matrix4::translation(_rotationPoint)*
                Matrix4::rotationX(-0.51_radf*delta.y())*
                Matrix4::rotationY(-0.51_radf*delta.x())*
                Matrix4::translation(-_rotationPoint));
    } else {
        const Vector3 p = unproject(event.position(), _lastDepth);
        _objCamera->translateLocal(_translationPoint - p); /* is Z always 0? */
        _translationPoint = p;
    }

    event.setAccepted();
}

void FluidSimApp::mouseScrollEvent(MouseScrollEvent& event) {
    const Float delta = event.offset().y();
    if(Math::abs(delta) < 1.0e-2f) {
        return;
    }

    //    if(_imGuiContext.handleMouseScrollEvent(event)) {
    //        /* Prevent scrolling the page */
    //        event.setAccepted();
    //        return;
    //    }

    const Float currentDepth = depthAt(event.position());
    const Float depth = currentDepth == 1.0f ? _lastDepth : currentDepth;
    const Vector3 p = unproject(event.position(), depth);
    /* Update the rotation point only if we're not zooming against infinite
       depth or if the original rotation point is not yet initialized */
    if(currentDepth != 1.0f || _rotationPoint.isZero()) {
        _rotationPoint = p;
        _lastDepth = depth;
    }

    /* Move towards/backwards the rotation point in cam coords */
    _objCamera->translateLocal(_rotationPoint * delta * 0.1f);
}

void FluidSimApp::textInputEvent(TextInputEvent& event) {
    //    if(_imGuiContext.handleTextInputEvent(event)) {
    //        event.setAccepted(true);
    //    }
}

Float FluidSimApp::depthAt(const Vector2i& windowPosition) {
    /* First scale the position from being relative to window size to being
       relative to framebuffer size as those two can be different on HiDPI
       systems */
    const Vector2i position = windowPosition*Vector2{framebufferSize()}/Vector2{windowSize()};
    const Vector2i fbPosition{position.x(), GL::defaultFramebuffer.viewport().sizeY() - position.y() - 1};

    GL::defaultFramebuffer.mapForRead(GL::DefaultFramebuffer::ReadAttachment::Front);
    Image2D data = GL::defaultFramebuffer.read(
            Range2Di::fromSize(fbPosition, Vector2i{1}).padded(Vector2i{2}),
            {GL::PixelFormat::DepthComponent, GL::PixelType::Float});

    return Math::min<Float>(Containers::arrayCast<const Float>(data.data()));
}

Vector3 FluidSimApp::unproject(const Vector2i& windowPosition, float depth) const {
    /* We have to take window size, not framebuffer size, since the position is
       in window coordinates and the two can be different on HiDPI systems */
    const Vector2i viewSize = windowSize();
    const Vector2i viewPosition = Vector2i{windowPosition.x(), viewSize.y() - windowPosition.y() - 1};
    const Vector3 in{2.0f*Vector2{viewPosition}/Vector2{viewSize} - Vector2{1.0f}, depth*2.0f - 1.0f};

    return _camera->projectionMatrix().inverted().transformPoint(in);
}



void FluidSimApp::initializeScene() {



}

int FluidSimApp::nonDisplayExec()
{
    for(currentStep = 0; currentStep < nSteps; ++currentStep) {
        simulationStep();
    }
    return 0;
}

int FluidSimApp::exec()
{
    if(display) {
        return GlfwApplication::exec();
    }
    else {
        return nonDisplayExec();
    }
}

void FluidSimApp::simulationStep() {
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


    engineStep();

    currentStep += 1;
}



// include some standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <time.h>
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



/* What to do if ENGINE_FLAGS was not defined? */
#ifndef ENGINE_FLAGS
#define ENGINE_FLAGS engine_flag_none
#endif
#ifndef CPU_TPS
#define CPU_TPS 2.67e+9
#endif

void engineStep() {

    //return;

    engine *e = engine_get();

    ticks tic, toc_step, toc_temp;

    double epot, ekin, v2, temp;

    int   k, cid, pid;

    double w;

    // take a step
    tic = getticks();

    //ENGINE_DUMP("pre step: ");

    if ( engine_step(e) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return ;
    }

    //ENGINE_DUMP("after step: ");

    toc_step = getticks();

    /* Check virtual/local ids. */
    /* for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
               for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
                   if ( e.s.cells[cid].parts[pid].id != e.s.cells[cid].parts[pid].vid )
                       printf( "main: inconsistent particle id/vid (%i/%i)!\n",
                           e.s.cells[cid].parts[pid].id, e.s.cells[cid].parts[pid].vid ); */

    /* Verify integrity of partlist. */
    /* for ( k = 0 ; k < nr_mols*3 ; k++ )
               if ( e.s.partlist[k]->id != k )
                   printf( "main: inconsistent particle id/partlist (%i/%i)!\n", e.s.partlist[k]->id, k );
           fflush(stdout); */


    // get the total COM-velocities and ekin
    epot = e->s.epot; ekin = 0.0;
#pragma omp parallel for schedule(static,100), private(cid,pid,k,v2), reduction(+:epot,ekin)
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
        for ( pid = 0 ; pid < e->s.cells[cid].count ; pid++ ) {
            for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                v2 += e->s.cells[cid].parts[pid].v[k] * e->s.cells[cid].parts[pid].v[k];
            ekin += 0.5 * 39.948 * v2;
        }
    }

    // compute the temperature and scaling
    temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * e->s.nr_parts );
    w = sqrt( 1.0 + 0.1 * ( e->temperature / temp - 1.0 ) );

    // scale the velocities

    /*
    if ( i < 10000 ) {
#pragma omp parallel for schedule(static,100), private(cid,pid,k), reduction(+:epot,ekin)
        for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
            for ( pid = 0 ; pid < e->s.cells[cid].count ; pid++ ) {
                for ( k = 0 ; k < 3 ; k++ )
                    e->s.cells[cid].parts[pid].v[k] *= w;
            }
        }
    }
     */

    toc_temp = getticks();

    printf("time:%i, epot:%e, ekin:%e, temp:%e, swaps:%i, stalls: %i %.3f %.3f %.3f ms\n",
            e->time,epot,ekin,temp,e->s.nr_swaps,e->s.nr_stalls,
            (double)(toc_temp-tic) * 1000 / CPU_TPS,
            (double)(toc_step-tic) * 1000 / CPU_TPS,
            (double)(toc_temp-toc_step) * 1000 / CPU_TPS);
    fflush(stdout);

    // print some particle data
    // printf("main: part 13322 is at [ %e , %e , %e ].\n",
    //     e.s.partlist[13322]->x[0], e.s.partlist[13322]->x[1], e.s.partlist[13322]->x[2]);
}



int initArgon (const Vector3 &origin, const Vector3 &dim,
        int nParticles, double dt, float temp ) {
    
    engine *e = engine_get();

    double length = dim[0] - origin[0];

    double L[] = { 0.1 * length , 0.1* length , 0.1*  length  };

    double x[3];

    double   cutoff = 0.1 * length;

    struct MxParticle pAr;
    struct MxPotential *pot_ArAr;

    int  k, cid, pid, nr_runners = 8;

    auto pos = fillCubeRandom(origin, dim, nParticles);

    ticks tic, toc;

    tic = getticks();

    double _origin[3];
    double _dim[3];
    for(int i = 0; i < 3; ++i) {
        _origin[i] = origin[i];
        _dim[i] = dim[i];
    }

    // initialize the engine
    printf("main: initializing the engine... ");
    printf("main: requesting origin = [ %f , %f , %f ].\n", _origin[0], _origin[1], _origin[2] );
    printf("main: requesting dimensions = [ %f , %f , %f ].\n", _dim[0], _dim[1], _dim[2] );
    printf("main: requesting cell size = [ %f , %f , %f ].\n", L[0], L[1], L[2] );
    printf("main: requesting cutoff = %22.16e.\n", cutoff);
    fflush(stdout);

    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init(e , _origin , _dim , L , cutoff , space_periodic_full , 2 , engine_flag_none ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }

    e->dt = dt;
    e->temperature = temp;


    printf("main: n_cells: %i, cell width set to %22.16e.\n", e->s.nr_cells, cutoff);

    printf("done.\n"); fflush(stdout);

    // set the interaction cutoff
    printf("main: cell dimensions = [ %i , %i , %i ].\n", e->s.cdim[0] , e->s.cdim[1] , e->s.cdim[2] );
    printf("main: cell size = [ %e , %e , %e ].\n" , e->s.h[0] , e->s.h[1] , e->s.h[2] );
    printf("main: cutoff set to %22.16e.\n", cutoff);
    printf("main: nr tasks: %i.\n",e->s.nr_tasks);

    /* mix-up the pair list just for kicks
    printf("main: shuffling the interaction pairs... "); fflush(stdout);
    srand(6178);
    for ( i = 0 ; i < e.s.nr_pairs ; i++ ) {
        j = rand() % e.s.nr_pairs;
        if ( i != j ) {
            cp = e.s.pairs[i];
            e.s.pairs[i] = e.s.pairs[j];
            e.s.pairs[j] = cp;
            }
        }
    printf("done.\n"); fflush(stdout); */


    // initialize the Ar-Ar potential
    if ( ( pot_ArAr = potential_create_LJ126( 0.275 , cutoff, 9.5075e-06 , 6.1545e-03 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_LJ126 failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
    }
    printf("main: constructed ArAr-potential with %i intervals.\n",pot_ArAr->n); fflush(stdout);


    /* register the particle types. */
    if ( ( pAr.typeId = engine_addtype(e , 39.948 , 0.0 , "Ar" , "Ar" ) ) < 0 ) {
        printf("main: call to engine_addtype failed.\n");
        errs_dump(stdout);
        return 1;
    }

    // register these potentials.
    if ( engine_addpot( e , pot_ArAr , pAr.typeId , pAr.typeId ) < 0 ){
        printf("main: call to engine_addpot failed.\n");
        errs_dump(stdout);
        return 1;
    }

    // set fields for all particles
    srand(6178);

    pAr.flags = PARTICLE_FLAG_NONE;
    for ( k = 0 ; k < 3 ; k++ ) {
        pAr.x[k] = 0.0;
        pAr.v[k] = 0.0;
        pAr.f[k] = 0.0;
    }

    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);

    // total velocity squared
    float totV2 = 0;

    for(int i = 0; i < pos.size(); ++i) {
        pAr.id = i;

        pAr.v[0] = ((double)rand()) / RAND_MAX - 0.5;
        pAr.v[1] = ((double)rand()) / RAND_MAX - 0.5;
        pAr.v[2] = ((double)rand()) / RAND_MAX - 0.5;

        totV2 +=   pAr.v[0]*pAr.v[0] + pAr.v[1]*pAr.v[1] + pAr.v[2]*pAr.v[2] ;

        x[0] = pos[i][0];
        x[1] = pos[i][1];
        x[2] = pos[i][2];

        if ( space_addpart( &(e->s) , &pAr , x, NULL ) != 0 ) {
            printf("main: space_addpart failed with space_err=%i.\n",space_err);
            errs_dump(stdout);
            return 1;
        }
    }

    float t = (1./ 3.) * e->types[pAr.typeId]->mass * totV2 / e->s.nr_parts;
    std::cout << "temperature before scaling: " << t << std::endl;

    float vScale = sqrt((3./e->types[pAr.typeId]->mass) * (e->temperature) / (totV2 / e->s.nr_parts));

    // sanity check
    totV2 = 0;

    // scale velocities
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
        for ( pid = 0 ; pid < e->s.cells[cid].count ; pid++ ) {
            for ( k = 0 ; k < 3 ; k++ ) {
                e->s.cells[cid].parts[pid].v[k] *= vScale;
                totV2 += e->s.cells[cid].parts[pid].v[k] * e->s.cells[cid].parts[pid].v[k];
            }
        }
    }

    t = (1./ 3.) * e->types[pAr.typeId]->mass * totV2 / e->s.nr_parts;
    std::cout << "particle temperature: " << t << std::endl;




    printf("done.\n"); fflush(stdout);
    printf("main: inserted %i particles.\n", e->s.nr_parts);

    // set the time and time-step by hand
    e->time = 0;

    printf("main: dt set to %f fs.\n", e->dt*1000 );

    toc = getticks();

    printf("main: setup took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS);



    // start the engine

    if ( engine_start(e , nr_runners , nr_runners ) != 0 ) {
        printf("main: engine_start failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
    }


    return 0;
}
void test() {

    glfwInit ();

    int majorVersion = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);

    std::cout << "major version: " << majorVersion << std::endl;
}

int main(int argc, char** argv) {

    //test();


    FluidSimApp app({argc, argv});
    return app.exec();
}





