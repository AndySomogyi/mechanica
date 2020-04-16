/*
 * MxSurfaceSimulator.h
 *
 *  Created on: Mar 28, 2019
 *      Author: andy
 */

#ifndef SRC_MXSURFACESIMULATOR_H_
#define SRC_MXSURFACESIMULATOR_H_

#include <Mechanica.h>
#include "mechanica_private.h"
#include "rendering/MxApplication.h"
#include "MxCylinderModel.h"
#include "LangevinPropagator.h"
#include <rendering/MxMeshRenderer.h>
#include <rendering/ArcBallInteractor.h>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>


struct MxSurfaceSimulator_Config {

    /**
     * Size of the frame buffer, width x height
     */
    int frameBufferSize[2];

    /**
     * path to model to load on initialization.
     */
    const char* modelPath;

    /**
     * Ignored if application already exists.
     */
    MxApplicationConfig applicationConfig;
};

struct MxSurfaceSimulator : CObject
{
    typedef MxSurfaceSimulator_Config Configuration;

    /**
     * Create a basic simulator
     */
    MxSurfaceSimulator(const Configuration &config);

    MxCylinderModel *model = nullptr;

    LangevinPropagator *propagator = nullptr;

    Magnum::Matrix4 transformation, projection;
    Magnum::Vector2 previousMousePosition;

    Magnum::Matrix4 rotation;

    Vector3 centerShift{0., 0., -18};


    Color4 color; // = Color4::fromHsv(color.hue() + 50.0_degf, 1.0f, 1.0f);
    Vector3 center;

    // distance from camera, move by mouse
    float distance = -3;


    MxMeshRenderer *renderer = nullptr;

    void loadModel(const char* fileName);

    void step(float dt);

    void draw();

    void mouseMove(double xpos, double ypos);

    void mouseClick(int button, int action, int mods);

    int timeSteps = 0;

    ArcBallInteractor arcBall;

    GL::Renderbuffer renderBuffer;

    GL::Framebuffer frameBuffer;

    HRESULT createContext(const Configuration& configuration);

};



MxSurfaceSimulator *MxSurfaceSimulator_New(const MxSurfaceSimulator_Config *conf);


HRESULT MxSurfaceSimulator_LoadModel(MxSurfaceSimulator *sim, const char* modelPath);



HRESULT MxSurfaceSimulator_init(PyObject *o);

PyObject *MxSurfaceSimulator_ImageData(MxSurfaceSimulator *self, const char* path);


#endif /* SRC_MXSURFACESIMULATOR_H_ */
