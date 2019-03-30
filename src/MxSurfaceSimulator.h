/*
 * MxSurfaceSimulator.h
 *
 *  Created on: Mar 28, 2019
 *      Author: andy
 */

#ifndef SRC_MXSURFACESIMULATOR_H_
#define SRC_MXSURFACESIMULATOR_H_

#include "mechanica_private.h"
#include "MxApplication.h"

struct MxSurfaceSimulator : MxObject
{

    struct Configuration {

    };

    /**
     * Create a basic simulator
     */
    MxSurfaceSimulator(const Configuration &config);

};



HRESULT MxSurfaceSimulator_init(PyObject *o);


#endif /* SRC_MXSURFACESIMULATOR_H_ */
