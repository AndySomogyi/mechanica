/*
 * MxParticles.cpp
 *
 *  Created on: Feb 25, 2017
 *      Author: andy
 */

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <MxUtil.h>
#include <MxPy.h>
#include "numpy/arrayobject.h"

static PyObject *random_point_disk(int n) {

    try {
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        int nd = 2;

        int typenum = NPY_DOUBLE;

        npy_intp dims[] = {n,3};

        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);

        double *data = (double*)PyArray_DATA(array);

        for(int i = 0; i < n; ++i) {
            double r = sqrt(uniform01(CRandom));
            double theta = 2 * M_PI * uniform01(CRandom);
            data[i * 3 + 0] = r * cos(theta);
            data[i * 3 + 1] = r * sin(theta);
            data[i * 3 + 2] = 0.;
        }

        return (PyObject*)array;

    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return NULL;
    }
}


static PyObject* random_point_sphere(int n) {

    try {

        double radius = 1.0;



        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        int nd = 2;

        int typenum = NPY_DOUBLE;

        npy_intp dims[] = {n,3};


        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);

        double *data = (double*)PyArray_DATA(array);

        for(int i = 0; i < n; ++i) {
            double theta = 2 * M_PI * uniform01(CRandom);
            double phi = acos(1 - 2 * uniform01(CRandom));
            double x = radius * sin(phi) * cos(theta);
            double y = radius * sin(phi) * sin(theta);
            double z = radius * cos(phi);

            data[i * 3 + 0] = x;
            data[i * 3 + 1] = y;
            data[i * 3 + 2] = z;
        }

        return (PyObject*)array;

    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return NULL;
    }
}


PyObject* MxPoints(PyObject *m, PyObject *args, PyObject *kwargs)
{
    try {
        MxPointsType kind = arg<MxPointsType>("kind", 0, args, kwargs, MxPointsType::Sphere);
        int n  = arg<int>("n", 1, args, kwargs, 1);

        switch(kind) {
        case MxPointsType::Sphere:
            return random_point_sphere(n);
        case MxPointsType::Disk:
            return random_point_disk(n);
        default:
            PyErr_SetString(PyExc_ValueError, "invalid kind");
            return NULL;
        }
    }
    catch (const std::exception &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return NULL;
    }
    catch(pybind11::error_already_set &e){
        e.restore();
        return NULL;
    }
}



HRESULT _MxUtil_init(PyObject *m)
{
    pybind11::enum_<MxPointsType>(m, "RandomPoints")
    .value("Sphere", MxPointsType::Sphere)
    .value("SolidSphere", MxPointsType::SolidSphere)
    .value("Disk", MxPointsType::Disk)
    .export_values();

    return S_OK;
}
