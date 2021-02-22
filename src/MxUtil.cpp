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
#include <MxNumpy.h>
#include <MxConvert.hpp>
#include <MxThreadPool.hpp>

#include <CConvert.hpp>
#include <mdcore_config.h>
#include <engine.h>

#include "Magnum/Mesh.h"
#include "Magnum/Math/Vector3.h"
#include "Magnum/MeshTools/RemoveDuplicates.h"
#include "Magnum/MeshTools/Subdivide.h"
#include "Magnum/Trade/ArrayAllocator.h"
#include "Magnum/Trade/MeshData.h"

#include "metrics.h"

#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <set>
#include <iostream>
#include <vector>
#include <bitset>
#include <array>
#include <string>

#include <MxPy.h>



const char* MxColor3Names[] = {
    "AliceBlue",
    "AntiqueWhite",
    "Aqua",
    "Aquamarine",
    "Azure",
    "Beige",
    "Bisque",
    "Black",
    "BlanchedAlmond",
    "Blue",
    "BlueViolet",
    "Brown",
    "BurlyWood",
    "CadetBlue",
    "Chartreuse",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkBlue",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGray",
    "DarkGreen",
    "DarkKhaki",
    "DarkMagenta",
    "DarkOliveGreen",
    "Darkorange",
    "DarkOrchid",
    "DarkRed",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkSlateBlue",
    "DarkSlateGray",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DimGray",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    "GoldenRod",
    "Gray",
    "Green",
    "GreenYellow",
    "HoneyDew",
    "HotPink",
    "IndianRed",
    "Indigo",
    "Ivory",
    "Khaki",
    "Lavender",
    "LavenderBlush",
    "LawnGreen",
    "LemonChiffon",
    "LightBlue",
    "LightCoral",
    "LightCyan",
    "LightGoldenRodYellow",
    "LightGrey",
    "LightGreen",
    "LightPink",
    "LightSalmon",
    "LightSeaGreen",
    "LightSkyBlue",
    "LightSlateGray",
    "LightSteelBlue",
    "LightYellow",
    "Lime",
    "LimeGreen",
    "Linen",
    "Magenta",
    "Maroon",
    "MediumAquaMarine",
    "MediumBlue",
    "MediumOrchid",
    "MediumPurple",
    "MediumSeaGreen",
    "MediumSlateBlue",
    "MediumSpringGreen",
    "MediumTurquoise",
    "MediumVioletRed",
    "MidnightBlue",
    "MintCream",
    "MistyRose",
    "Moccasin",
    "NavajoWhite",
    "Navy",
    "OldLace",
    "Olive",
    "OliveDrab",
    "Orange",
    "OrangeRed",
    "Orchid",
    "PaleGoldenRod",
    "PaleGreen",
    "PaleTurquoise",
    "PaleVioletRed",
    "PapayaWhip",
    "PeachPuff",
    "Peru",
    "Pink",
    "Plum",
    "PowderBlue",
    "Purple",
    "Red",
    "RosyBrown",
    "RoyalBlue",
    "SaddleBrown",
    "Salmon",
    "SandyBrown",
    "SeaGreen",
    "SeaShell",
    "Sienna",
    "Silver",
    "SkyBlue",
    "SlateBlue",
    "SlateGray",
    "Snow",
    "SpringGreen",
    "SteelBlue",
    "Tan",
    "Teal",
    "Thistle",
    "Tomato",
    "Turquoise",
    "Violet",
    "Wheat",
    "White",
    "WhiteSmoke",
    "Yellow",
    "YellowGreen",
    "SpaBlue",
    "Pumpkin",
    "OleumYellow",
    "SGIPurple",
    NULL
};

typedef float (*force_2body_fn)(struct EnergyMinimizer* p, Magnum::Vector3 *x1, Magnum::Vector3 *x2,
Magnum::Vector3 *f1, Magnum::Vector3 *f2);

typedef float (*force_1body_fn)(struct EnergyMinimizer* p, Magnum::Vector3 *p1,
Magnum::Vector3 *f1);

struct EnergyMinimizer {
    force_1body_fn force_1body;
    force_2body_fn force_2body;
    int max_outer_iter;
    int max_inner_iter;
    float outer_de;
    float inner_de;
    float cutoff;
};

static void energy_minimize(EnergyMinimizer *p, std::vector<Magnum::Vector3> &points);

static float sphere_2body(EnergyMinimizer* p, Magnum::Vector3 *x1, Magnum::Vector3 *x2,
                          Magnum::Vector3 *f1, Magnum::Vector3 *f2);

static float sphere_1body(EnergyMinimizer* p, Magnum::Vector3 *p1,
                          Magnum::Vector3 *f1) ;

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
        C_EXP(e); return NULL;
    }
}


static PyObject* random_point_sphere(int n) {

    try {
        std::vector<Magnum::Vector3> points(n);
        
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

            points[i] = Magnum::Vector3{(float)x, (float)y, (float)z};
        }
        
        EnergyMinimizer em;
        em.force_1body = sphere_1body;
        em.force_2body = sphere_2body;
        em.max_outer_iter = 10;
        em.cutoff = 0.2;
        
        energy_minimize(&em, points);
        
        for(int i = 0; i < n; ++i) {
            data[i * 3 + 0] = points[i].x();
            data[i * 3 + 1] = points[i].y();
            data[i * 3 + 2] = points[i].z();
        }

        return (PyObject*)array;

    }
    catch (const std::exception &e) {
        C_EXP(e); return NULL;
    }
}

static PyObject* random_point_solidsphere(int n) {

    try {

        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        int nd = 2;

        int typenum = NPY_DOUBLE;

        npy_intp dims[] = {n,3};


        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);

        double *data = (double*)PyArray_DATA(array);

        for(int i = 0; i < n; ++i) {
            double theta = 2 * M_PI * uniform01(CRandom);
            double phi = acos(1 - 2 * uniform01(CRandom));
            double r = std::cbrt(uniform01(CRandom));
            double x = r * sin(phi) * cos(theta);
            double y = r * sin(phi) * sin(theta);
            double z = r * cos(phi);

            data[i * 3 + 0] = x;
            data[i * 3 + 1] = y;
            data[i * 3 + 2] = z;
        }

        return (PyObject*)array;

    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}

static PyObject* random_point_solidsphere_shell(int n, PyObject *_dr, PyObject *_phi) {
    float dr = 1.0;
    
    if(_dr) {
        dr = mx::cast<float>(_dr);
    }
    
    float phi0 = 0;
    float phi1 = M_PI;
    
    if(_phi) {
        if(!PyTuple_Check(_phi) || PyTuple_Size(_phi) != 2) {
            throw std::logic_error("phi must be a tuple of (phi0, phi1)");
        }
        phi0 = mx::cast<float>(PyTuple_GET_ITEM(_phi, 0));
        phi1 = mx::cast<float>(PyTuple_GET_ITEM(_phi, 1));
    }
    

    double cos0 = std::cos(phi0);
    double cos1 = std::cos(phi1);
    
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    
    int nd = 2;
    
    int typenum = NPY_DOUBLE;
    
    npy_intp dims[] = {n,3};
    
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);
    
    double *data = (double*)PyArray_DATA(array);
    
    for(int i = 0; i < n; ++i) {
        double theta = 2 * M_PI * uniform01(CRandom);
        // double phi = acos(1 - 2 * uniform01(CRandom));
        double phi = acos(cos0 - (cos0-cos1) * uniform01(CRandom));
        double r = std::cbrt((1-dr) + dr * uniform01(CRandom));
        double x = r * sin(phi) * cos(theta);
        double y = r * sin(phi) * sin(theta);
        double z = r * cos(phi);
        
        data[i * 3 + 0] = x;
        data[i * 3 + 1] = y;
        data[i * 3 + 2] = z;
    }
    
    return (PyObject*)array;
}

static PyObject* random_point_solidcube(int n) {

    try {
        std::uniform_real_distribution<double> uniform01(-0.5, 0.5);

        int nd = 2;

        int typenum = NPY_DOUBLE;

        npy_intp dims[] = {n,3};

        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);

        double *data = (double*)PyArray_DATA(array);

        for(int i = 0; i < n; ++i) {
            double x = uniform01(CRandom);
            double y = uniform01(CRandom);
            double z = uniform01(CRandom);
            data[i * 3 + 0] = x;
            data[i * 3 + 1] = y;
            data[i * 3 + 2] = z;
        }

        return (PyObject*)array;

    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}

static PyObject* points_solidcube(int n) {
    
    if(n < 8) {
        PyErr_SetString(PyExc_ValueError, "minimum 8 points in cube");
        return NULL;
    }
    
    try {
        std::uniform_real_distribution<double> uniform01(-0.5, 0.5);
        
        int nd = 2;
        
        int typenum = NPY_DOUBLE;
        
        npy_intp dims[] = {n,3};
        
        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);
        
        double *data = (double*)PyArray_DATA(array);
        
        for(int i = 0; i < n; ++i) {
            double x = uniform01(CRandom);
            double y = uniform01(CRandom);
            double z = uniform01(CRandom);
            data[i * 3 + 0] = x;
            data[i * 3 + 1] = y;
            data[i * 3 + 2] = z;
        }
        
        return (PyObject*)array;
        
    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}



static PyObject* points_ring(int n) {
    try {
        double radius = 1.0;
        
        int nd = 2;
        
        int typenum = NPY_DOUBLE;
        
        npy_intp dims[] = {n,3};
        
        const double phi = M_PI / 2.;
        const double theta_i = 2 * M_PI / n;
        
        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);
        
        double *data = (double*)PyArray_DATA(array);
        
        for(int i = 0; i < n; ++i) {
            double theta = i * theta_i;
            double x = radius * sin(phi) * cos(theta);
            double y = radius * sin(phi) * sin(theta);
            double z = radius * cos(phi);
            
            data[i * 3 + 0] = x;
            data[i * 3 + 1] = y;
            data[i * 3 + 2] = z;
        }
        
        return (PyObject*)array;
        
    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}

static PyObject* points_sphere(int n) {
    
    std::vector<Magnum::Vector3> vertices;
    std::vector<int32_t> indices;
    Mx_Icosphere(n, 0, M_PI, vertices, indices);
    
    
    try {
        int nd = 2;
        npy_intp dims[] = {(int)vertices.size(),3};
        int typenum = NPY_DOUBLE;
        
        PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(nd, dims, typenum);
        
        double *data = (double*)PyArray_DATA(array);
        
        for(int i = 0; i < vertices.size(); ++i) {
            Magnum::Vector3 vec = vertices[i];
            
            
            data[i * 3 + 0] = vec.x();
            data[i * 3 + 1] = vec.y();
            data[i * 3 + 2] = vec.z();
        }
        
        return (PyObject*)array;
        
    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}




PyObject* MxRandomPoints(PyObject *m, PyObject *args, PyObject *kwargs)
{
    try {
        MxPointsType kind = (MxPointsType)mx::arg<int>("kind", 0, args, kwargs, MxPointsType::Sphere);
        int n  = mx::arg<int>("n", 1, args, kwargs, 1);

        switch(kind) {
        case MxPointsType::Sphere:
            return random_point_sphere(n);
        case MxPointsType::Disk:
            return random_point_disk(n);
        case MxPointsType::SolidCube:
            return random_point_solidcube(n);
        case MxPointsType::SolidSphere: {
            PyObject *phi = mx::py_arg("phi", 2, args, kwargs);
            PyObject *dr = mx::py_arg("dr", 3, args, kwargs);
            return random_point_solidsphere_shell(n, dr, phi);
        }
        default:
            PyErr_SetString(PyExc_ValueError, "invalid kind");
            return NULL;
        }
    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}


PyObject* MxPoints(PyObject *m, PyObject *args, PyObject *kwargs)
{
    try {
        MxPointsType kind = (MxPointsType)mx::arg<int>("kind", 0, args, kwargs, MxPointsType::Sphere);
        int n  = mx::arg<int>("n", 1, args, kwargs, 1);
        
        switch(kind) {
            case MxPointsType::Ring:
                return points_ring(n);
            case MxPointsType::Sphere:
                return points_sphere(n);
            default:
                PyErr_SetString(PyExc_ValueError, "invalid kind");
                return NULL;
        }
    }
    catch (const std::exception& e) {
        C_EXP(e); return NULL;
    }
}



HRESULT _MxUtil_init(PyObject *m)
{
    PyModule_AddIntConstant(m, "Sphere", MxPointsType::Sphere);
    PyModule_AddIntConstant(m, "SolidSphere", MxPointsType::SolidSphere);
    PyModule_AddIntConstant(m, "Disk", MxPointsType::Disk);
    PyModule_AddIntConstant(m, "SolidCube", MxPointsType::SolidCube);
    PyModule_AddIntConstant(m, "Cube", MxPointsType::Cube);
    PyModule_AddIntConstant(m, "Ring", MxPointsType::Ring);

    return S_OK;
}

Magnum::Color3 Color3_Parse(const std::string &s)
{
    if(s.length() < 2) {
        // TODO ???
        return Magnum::Color3{};
    }

    // #ff6347
    if(s.length() >= 0 && s[0] == '#') {
        std::string srgb = s.substr(1, s.length() - 1);

        char* ss;
        unsigned long rgb = strtoul(srgb.c_str(), &ss, 16);

        return Magnum::Color3::fromSrgb(rgb);
    }

    std::string str = s;
    std::transform(str.begin(), str.end(),str.begin(), ::toupper);

    // TODO, thread safe???
    static std::unordered_map<std::string, Magnum::Color3> colors;
    if(colors.size() == 0) {
        colors["INDIANRED"]         = Magnum::Color3::fromSrgb(0xCD5C5C);
        colors["LIGHTCORAL"]        = Magnum::Color3::fromSrgb(0xF08080);
        colors["SALMON"]            = Magnum::Color3::fromSrgb(0xFA8072);
        colors["DARKSALMON"]        = Magnum::Color3::fromSrgb(0xE9967A);
        colors["LIGHTSALMON"]       = Magnum::Color3::fromSrgb(0xFFA07A);
        colors["CRIMSON"]           = Magnum::Color3::fromSrgb(0xDC143C);
        colors["RED"]               = Magnum::Color3::fromSrgb(0xFF0000);
        colors["FIREBRICK"]         = Magnum::Color3::fromSrgb(0xB22222);
        colors["DARKRED"]           = Magnum::Color3::fromSrgb(0x8B0000);
        colors["PINK"]              = Magnum::Color3::fromSrgb(0xFFC0CB);
        colors["LIGHTPINK"]         = Magnum::Color3::fromSrgb(0xFFB6C1);
        colors["HOTPINK"]           = Magnum::Color3::fromSrgb(0xFF69B4);
        colors["DEEPPINK"]          = Magnum::Color3::fromSrgb(0xFF1493);
        colors["MEDIUMVIOLETRED"]   = Magnum::Color3::fromSrgb(0xC71585);
        colors["PALEVIOLETRED"]     = Magnum::Color3::fromSrgb(0xDB7093);
        colors["LIGHTSALMON"]       = Magnum::Color3::fromSrgb(0xFFA07A);
        colors["CORAL"]             = Magnum::Color3::fromSrgb(0xFF7F50);
        colors["TOMATO"]            = Magnum::Color3::fromSrgb(0xFF6347);
        colors["ORANGERED"]         = Magnum::Color3::fromSrgb(0xFF4500);
        colors["DARKORANGE"]        = Magnum::Color3::fromSrgb(0xFF8C00);
        colors["ORANGE"]            = Magnum::Color3::fromSrgb(0xFFA500);
        colors["GOLD"]              = Magnum::Color3::fromSrgb(0xFFD700);
        colors["YELLOW"]            = Magnum::Color3::fromSrgb(0xFFFF00);
        colors["LIGHTYELLOW"]       = Magnum::Color3::fromSrgb(0xFFFFE0);
        colors["LEMONCHIFFON"]      = Magnum::Color3::fromSrgb(0xFFFACD);
        colors["LIGHTGOLDENRODYELLOW"] = Magnum::Color3::fromSrgb(0xFAFAD2);
        colors["PAPAYAWHIP"]        = Magnum::Color3::fromSrgb(0xFFEFD5);
        colors["MOCCASIN"]          = Magnum::Color3::fromSrgb(0xFFE4B5);
        colors["PEACHPUFF"]         = Magnum::Color3::fromSrgb(0xFFDAB9);
        colors["PALEGOLDENROD"]     = Magnum::Color3::fromSrgb(0xEEE8AA);
        colors["KHAKI"]             = Magnum::Color3::fromSrgb(0xF0E68C);
        colors["DARKKHAKI"]         = Magnum::Color3::fromSrgb(0xBDB76B);
        colors["LAVENDER"]          = Magnum::Color3::fromSrgb(0xE6E6FA);
        colors["THISTLE"]           = Magnum::Color3::fromSrgb(0xD8BFD8);
        colors["PLUM"]              = Magnum::Color3::fromSrgb(0xDDA0DD);
        colors["VIOLET"]            = Magnum::Color3::fromSrgb(0xEE82EE);
        colors["ORCHID"]            = Magnum::Color3::fromSrgb(0xDA70D6);
        colors["FUCHSIA"]           = Magnum::Color3::fromSrgb(0xFF00FF);
        colors["MAGENTA"]           = Magnum::Color3::fromSrgb(0xFF00FF);
        colors["MEDIUMORCHID"]      = Magnum::Color3::fromSrgb(0xBA55D3);
        colors["MEDIUMPURPLE"]      = Magnum::Color3::fromSrgb(0x9370DB);
        colors["REBECCAPURPLE"]     = Magnum::Color3::fromSrgb(0x663399);
        colors["BLUEVIOLET"]        = Magnum::Color3::fromSrgb(0x8A2BE2);
        colors["DARKVIOLET"]        = Magnum::Color3::fromSrgb(0x9400D3);
        colors["DARKORCHID"]        = Magnum::Color3::fromSrgb(0x9932CC);
        colors["DARKMAGENTA"]       = Magnum::Color3::fromSrgb(0x8B008B);
        colors["PURPLE"]            = Magnum::Color3::fromSrgb(0x800080);
        colors["INDIGO"]            = Magnum::Color3::fromSrgb(0x4B0082);
        colors["SLATEBLUE"]         = Magnum::Color3::fromSrgb(0x6A5ACD);
        colors["DARKSLATEBLUE"]     = Magnum::Color3::fromSrgb(0x483D8B);
        colors["MEDIUMSLATEBLUE"]   = Magnum::Color3::fromSrgb(0x7B68EE);
        colors["GREENYELLOW"]       = Magnum::Color3::fromSrgb(0xADFF2F);
        colors["CHARTREUSE"]        = Magnum::Color3::fromSrgb(0x7FFF00);
        colors["LAWNGREEN"]         = Magnum::Color3::fromSrgb(0x7CFC00);
        colors["LIME"]              = Magnum::Color3::fromSrgb(0x00FF00);
        colors["LIMEGREEN"]         = Magnum::Color3::fromSrgb(0x32CD32);
        colors["PALEGREEN"]         = Magnum::Color3::fromSrgb(0x98FB98);
        colors["LIGHTGREEN"]        = Magnum::Color3::fromSrgb(0x90EE90);
        colors["MEDIUMSPRINGGREEN"] = Magnum::Color3::fromSrgb(0x00FA9A);
        colors["SPRINGGREEN"]       = Magnum::Color3::fromSrgb(0x00FF7F);
        colors["MEDIUMSEAGREEN"]    = Magnum::Color3::fromSrgb(0x3CB371);
        colors["SEAGREEN"]          = Magnum::Color3::fromSrgb(0x2E8B57);
        colors["FORESTGREEN"]       = Magnum::Color3::fromSrgb(0x228B22);
        colors["GREEN"]             = Magnum::Color3::fromSrgb(0x008000);
        colors["DARKGREEN"]         = Magnum::Color3::fromSrgb(0x006400);
        colors["YELLOWGREEN"]       = Magnum::Color3::fromSrgb(0x9ACD32);
        colors["OLIVEDRAB"]         = Magnum::Color3::fromSrgb(0x6B8E23);
        colors["OLIVE"]             = Magnum::Color3::fromSrgb(0x808000);
        colors["DARKOLIVEGREEN"]    = Magnum::Color3::fromSrgb(0x556B2F);
        colors["MEDIUMAQUAMARINE"]  = Magnum::Color3::fromSrgb(0x66CDAA);
        colors["DARKSEAGREEN"]      = Magnum::Color3::fromSrgb(0x8FBC8B);
        colors["LIGHTSEAGREEN"]     = Magnum::Color3::fromSrgb(0x20B2AA);
        colors["DARKCYAN"]          = Magnum::Color3::fromSrgb(0x008B8B);
        colors["TEAL"]              = Magnum::Color3::fromSrgb(0x008080);
        colors["AQUA"]              = Magnum::Color3::fromSrgb(0x00FFFF);
        colors["CYAN"]              = Magnum::Color3::fromSrgb(0x00FFFF);
        colors["LIGHTCYAN"]         = Magnum::Color3::fromSrgb(0xE0FFFF);
        colors["PALETURQUOISE"]     = Magnum::Color3::fromSrgb(0xAFEEEE);
        colors["AQUAMARINE"]        = Magnum::Color3::fromSrgb(0x7FFFD4);
        colors["TURQUOISE"]         = Magnum::Color3::fromSrgb(0x40E0D0);
        colors["MEDIUMTURQUOISE"]   = Magnum::Color3::fromSrgb(0x48D1CC);
        colors["DARKTURQUOISE"]     = Magnum::Color3::fromSrgb(0x00CED1);
        colors["CADETBLUE"]         = Magnum::Color3::fromSrgb(0x5F9EA0);
        colors["STEELBLUE"]         = Magnum::Color3::fromSrgb(0x4682B4);
        colors["LIGHTSTEELBLUE"]    = Magnum::Color3::fromSrgb(0xB0C4DE);
        colors["POWDERBLUE"]        = Magnum::Color3::fromSrgb(0xB0E0E6);
        colors["LIGHTBLUE"]         = Magnum::Color3::fromSrgb(0xADD8E6);
        colors["SKYBLUE"]           = Magnum::Color3::fromSrgb(0x87CEEB);
        colors["LIGHTSKYBLUE"]      = Magnum::Color3::fromSrgb(0x87CEFA);
        colors["DEEPSKYBLUE"]       = Magnum::Color3::fromSrgb(0x00BFFF);
        colors["DODGERBLUE"]        = Magnum::Color3::fromSrgb(0x1E90FF);
        colors["CORNFLOWERBLUE"]    = Magnum::Color3::fromSrgb(0x6495ED);
        colors["MEDIUMSLATEBLUE"]   = Magnum::Color3::fromSrgb(0x7B68EE);
        colors["ROYALBLUE"]         = Magnum::Color3::fromSrgb(0x4169E1);
        colors["BLUE"]              = Magnum::Color3::fromSrgb(0x0000FF);
        colors["MEDIUMBLUE"]        = Magnum::Color3::fromSrgb(0x0000CD);
        colors["DARKBLUE"]          = Magnum::Color3::fromSrgb(0x00008B);
        colors["NAVY"]              = Magnum::Color3::fromSrgb(0x000080);
        colors["MIDNIGHTBLUE"]      = Magnum::Color3::fromSrgb(0x191970);
        colors["CORNSILK"]          = Magnum::Color3::fromSrgb(0xFFF8DC);
        colors["BLANCHEDALMOND"]    = Magnum::Color3::fromSrgb(0xFFEBCD);
        colors["BISQUE"]            = Magnum::Color3::fromSrgb(0xFFE4C4);
        colors["NAVAJOWHITE"]       = Magnum::Color3::fromSrgb(0xFFDEAD);
        colors["WHEAT"]             = Magnum::Color3::fromSrgb(0xF5DEB3);
        colors["BURLYWOOD"]         = Magnum::Color3::fromSrgb(0xDEB887);
        colors["TAN"]               = Magnum::Color3::fromSrgb(0xD2B48C);
        colors["ROSYBROWN"]         = Magnum::Color3::fromSrgb(0xBC8F8F);
        colors["SANDYBROWN"]        = Magnum::Color3::fromSrgb(0xF4A460);
        colors["GOLDENROD"]         = Magnum::Color3::fromSrgb(0xDAA520);
        colors["DARKGOLDENROD"]     = Magnum::Color3::fromSrgb(0xB8860B);
        colors["PERU"]              = Magnum::Color3::fromSrgb(0xCD853F);
        colors["CHOCOLATE"]         = Magnum::Color3::fromSrgb(0xD2691E);
        colors["SADDLEBROWN"]       = Magnum::Color3::fromSrgb(0x8B4513);
        colors["SIENNA"]            = Magnum::Color3::fromSrgb(0xA0522D);
        colors["BROWN"]             = Magnum::Color3::fromSrgb(0xA52A2A);
        colors["MAROON"]            = Magnum::Color3::fromSrgb(0x800000);
        colors["WHITE"]             = Magnum::Color3::fromSrgb(0xFFFFFF);
        colors["SNOW"]              = Magnum::Color3::fromSrgb(0xFFFAFA);
        colors["HONEYDEW"]          = Magnum::Color3::fromSrgb(0xF0FFF0);
        colors["MINTCREAM"]         = Magnum::Color3::fromSrgb(0xF5FFFA);
        colors["AZURE"]             = Magnum::Color3::fromSrgb(0xF0FFFF);
        colors["ALICEBLUE"]         = Magnum::Color3::fromSrgb(0xF0F8FF);
        colors["GHOSTWHITE"]        = Magnum::Color3::fromSrgb(0xF8F8FF);
        colors["WHITESMOKE"]        = Magnum::Color3::fromSrgb(0xF5F5F5);
        colors["SEASHELL"]          = Magnum::Color3::fromSrgb(0xFFF5EE);
        colors["BEIGE"]             = Magnum::Color3::fromSrgb(0xF5F5DC);
        colors["OLDLACE"]           = Magnum::Color3::fromSrgb(0xFDF5E6);
        colors["FLORALWHITE"]       = Magnum::Color3::fromSrgb(0xFFFAF0);
        colors["IVORY"]             = Magnum::Color3::fromSrgb(0xFFFFF0);
        colors["ANTIQUEWHITE"]      = Magnum::Color3::fromSrgb(0xFAEBD7);
        colors["LINEN"]             = Magnum::Color3::fromSrgb(0xFAF0E6);
        colors["LAVENDERBLUSH"]     = Magnum::Color3::fromSrgb(0xFFF0F5);
        colors["MISTYROSE"]         = Magnum::Color3::fromSrgb(0xFFE4E1);
        colors["GAINSBORO"]         = Magnum::Color3::fromSrgb(0xDCDCDC);
        colors["LIGHTGRAY"]         = Magnum::Color3::fromSrgb(0xD3D3D3);
        colors["SILVER"]            = Magnum::Color3::fromSrgb(0xC0C0C0);
        colors["DARKGRAY"]          = Magnum::Color3::fromSrgb(0xA9A9A9);
        colors["GRAY"]              = Magnum::Color3::fromSrgb(0x808080);
        colors["DIMGRAY"]           = Magnum::Color3::fromSrgb(0x696969);
        colors["LIGHTSLATEGRAY"]    = Magnum::Color3::fromSrgb(0x778899);
        colors["SLATEGRAY"]         = Magnum::Color3::fromSrgb(0x708090);
        colors["DARKSLATEGRAY"]     = Magnum::Color3::fromSrgb(0x2F4F4F);
        colors["BLACK"]             = Magnum::Color3::fromSrgb(0x000000);
	colors["SPABLUE"]           = Magnum::Color3::fromSrgb(0x6D99D3); // Rust Oleum Spa Blue
	colors["PUMPKIN"]           = Magnum::Color3::fromSrgb(0xF65917); // Rust Oleum Pumpkin
	colors["OLEUMYELLOW"]       = Magnum::Color3::fromSrgb(0xF9CB20); // rust oleum yellow
	colors["SGIPURPLE"]         = Magnum::Color3::fromSrgb(0x6353BB); // SGI purple
    }

    std::unordered_map<std::string, Magnum::Color3>::const_iterator got =
            colors.find (str);

    if (got != colors.end()) {
        return got->second;
    }
    
    std::string warning = std::string("Warning, \"") + s + "\" is not a valid color name.";
    
    PyErr_WarnEx(PyExc_Warning, warning.c_str(), 0);

    return Magnum::Color3{};
}

constexpr uint32_t Indices[]{
    1, 2, 6,
    1, 7, 2,
    3, 4, 5,
    4, 3, 8,
    6, 5, 11,
    
    5, 6, 10,
    9, 10, 2,
    10, 9, 3,
    7, 8, 9,
    8, 7, 0,
    
    11, 0, 1,
    0, 11, 4,
    6, 2, 10,
    1, 6, 11,
    3, 5, 10,
    
    5, 4, 11,
    2, 7, 9,
    7, 1, 0,
    3, 9, 8,
    4, 8, 0
};

/* Can't be just an array of Vector3 because MSVC 2015 is special. See
 Crosshair.cpp for details. */
constexpr struct VertexSolidStrip {
    Magnum::Vector3 position;
} Vertices[] {
    {{0.0f, -0.525731f, 0.850651f}},
    {{0.850651f, 0.0f, 0.525731f}},
    {{0.850651f, 0.0f, -0.525731f}},
    {{-0.850651f, 0.0f, -0.525731f}},
    {{-0.850651f, 0.0f, 0.525731f}},
    {{-0.525731f, 0.850651f, 0.0f}},
    {{0.525731f, 0.850651f, 0.0f}},
    {{0.525731f, -0.850651f, 0.0f}},
    {{-0.525731f, -0.850651f, 0.0f}},
    {{0.0f, -0.525731f, -0.850651f}},
    {{0.0f, 0.525731f, -0.850651f}},
    {{0.0f, 0.525731f, 0.850651f}}
};

HRESULT Mx_Icosphere(const int subdivisions, float phi0, float phi1,
                    std::vector<Magnum::Vector3> &verts,
                    std::vector<int32_t> &inds) {
    
    // TODO: sloppy ass code, needs clean up...
    // total waste computing a whole sphere and throwign large parts away.
    
    const std::size_t indexCount =
        Magnum::Containers::arraySize(Indices) * (1 << subdivisions * 2);
    
    const std::size_t vertexCount =
        Magnum::Containers::arraySize(Vertices) +
        ((indexCount - Magnum::Containers::arraySize(Indices))/3);
    
    Magnum::Containers::Array<char> indexData{indexCount*sizeof(uint32_t)};
    
    auto indices = Magnum::Containers::arrayCast<uint32_t>(indexData);
    
    std::memcpy(indices.begin(), Indices, sizeof(Indices));
    
    struct Vertex {
        Magnum::Vector3 position;
        Magnum::Vector3 normal;
    };
    
    Magnum::Containers::Array<char> vertexData;
    Magnum::Containers::arrayResize<Magnum::Trade::ArrayAllocator>(
        vertexData, Magnum::Containers::NoInit, sizeof(Vertex)*vertexCount);
    
    /* Build up the subdivided positions */
    {
        auto vertices = Magnum::Containers::arrayCast<Vertex>(vertexData);
        Magnum::Containers::StridedArrayView1D<Magnum::Vector3>
            positions{vertices, &vertices[0].position, vertices.size(), sizeof(Vertex)};
        
        for(std::size_t i = 0; i != Magnum::Containers::arraySize(Vertices); ++i)
            positions[i] = Vertices[i].position;
        
        for(std::size_t i = 0; i != subdivisions; ++i) {
            const std::size_t iterationIndexCount =
                Magnum::Containers::arraySize(Indices)*(1 << (i + 1)*2);
            
            const std::size_t iterationVertexCount =
                Magnum::Containers::arraySize(Vertices) +
                ((iterationIndexCount - Magnum::Containers::arraySize(Indices))/3);
            
            Magnum::MeshTools::subdivideInPlace(
                indices.prefix(iterationIndexCount),
                positions.prefix(iterationVertexCount),
                [](const Magnum::Vector3& a, const Magnum::Vector3& b) {
                    return (a+b).normalized();
                }
            );
        }
        
        /** @todo i need arrayShrinkAndGiveUpMemoryIfItDoesntCauseRealloc() */
        Magnum::Containers::arrayResize<Magnum::Trade::ArrayAllocator>(
            vertexData,
            Magnum::MeshTools::removeDuplicatesIndexedInPlace(
                Magnum::Containers::stridedArrayView(indices),
                Magnum::Containers::arrayCast<2, char>(positions))*sizeof(Vertex)
        );
    }
    
    /* Build up the views again with correct size, fill the normals */
    auto vertices = Magnum::Containers::arrayCast<Vertex>(vertexData);
    
    Magnum::Containers::StridedArrayView1D<Magnum::Vector3>
        positions{vertices, &vertices[0].position, vertices.size(), sizeof(Vertex)};
    
    /*
     * original code
    verts.resize(positions.size());
    inds.resize(indices.size());
    
    for(int i = 0; i < positions.size(); ++i) {
        verts[i] = positions[i];
    }
    
    for(int i = 0; i < indices.size(); ++i) {
        inds[i] = indices[i];
    }
     */
    
    // prune the top and bottom vertices 
    verts.reserve(positions.size());
    inds.reserve(indices.size());
    
    std::vector<int32_t> index_map;
    index_map.resize(positions.size());
    
    std::set<int32_t> discarded_verts;
    
    Magnum::Vector3 origin = {0.0, 0.0, 0.0};
    for(int i = 0; i < positions.size(); ++i) {
        Magnum::Vector3 position = positions[i];
        Magnum::Vector3 spherical = MxCartesianToSpherical(position, origin);
        if(spherical[2] < phi0 || spherical[2] > phi1) {
            discarded_verts.emplace(i);
            index_map[i] = -1;
        }
        else {
            index_map[i] = verts.size();
            verts.push_back(position);
        }
    }
    
    for(int i = 0; i < indices.size(); i += 3) {
        int a = indices[i];
        int b = indices[i+1];
        int c = indices[i+2];
        
        if(discarded_verts.find(a) == discarded_verts.end() &&
           discarded_verts.find(b) == discarded_verts.end() &&
           discarded_verts.find(c) == discarded_verts.end()) {
            a = index_map[a];
            b = index_map[b];
            c = index_map[c];
            assert(a >= 0 && b >= 0 && c >= 0);
            inds.push_back(a);
            inds.push_back(b);
            inds.push_back(c);
        }
    }

    return S_OK;
}

static void energy_find_neighborhood(std::vector<Magnum::Vector3> const &points,
                                     const int part,
                                     float r,
                                     std::vector<int32_t> &nbor_inds,
                                     std::vector<int32_t> &boundary_inds) {
    
    nbor_inds.resize(0);
    boundary_inds.resize(0);
  
    float r2 = r * r;
    float br2 = 4 * r * r;
    
    Magnum::Vector3 pt = points[part];
    for(int i = 0; i < points.size(); ++i) {

        Magnum::Vector3 dx = pt - points[i];
        float dx2 = dx.dot();
        if(dx2 <= r2) {
            nbor_inds.push_back(i);
        }
        if(dx2 > r2 && dx2 <= br2) {
            boundary_inds.push_back(i);
        }
    }
}

float energy_minimize_neighborhood(EnergyMinimizer *p,
                                   std::vector<int32_t> &indices,
                                   std::vector<int32_t> &boundary_indices,
                                   std::vector<Magnum::Vector3> &points,
                                   std::vector<Magnum::Vector3> &forces) {

    float etot = 0;

    for(int i = 0; i < 10; i++) {
    
        float e = 0;
        
        for(int j = 0; j < indices.size(); ++j) {
            forces[indices[j]] = {0.0f, 0.0f, 0.0f};
        }
        
        for(int j = 0; j < indices.size(); ++j) {
            int32_t jj = indices[j];
            // one-body force
            e += p->force_1body(p, &points[jj], &forces[jj]);
            
            // two-body force in local neighborhood
            for(int k = j+1; k < indices.size(); ++k) {
                int32_t kk = indices[k];
                e += p->force_2body(p, &points[jj], &points[kk], &forces[jj], &forces[kk]);
            }
            
            // two-body force from boundaries
            for(int k = j+1; k < boundary_indices.size(); ++k) {
                int32_t kk = boundary_indices[k];
                e += p->force_2body(p, &points[jj], &points[kk], &forces[jj], nullptr) / 2;
            }
        }
        
        for(int i = 0; i < indices.size(); ++i) {
            int32_t ii = indices[i];
            points[ii] += 1 * forces[ii];
        }
        
        etot += e;
    }
    return etot;
}

void energy_minimize(EnergyMinimizer *p, std::vector<Magnum::Vector3> &points) {
    std::vector<int32_t> nindices(points.size()/2);
    std::vector<int32_t> bindices(points.size()/2);
    std::vector<Magnum::Vector3> forces(points.size());
    std::uniform_int_distribution<> distrib(0, points.size() - 1);
    
    float etot[3] = {0, 0, 0};
    float de[3] = {0, 0, 0};
    int ntot = 0;
    float de_avg = 0;
    int i = 0;
    
    do {
        for(int k = 0; k < points.size(); ++k) {
            int32_t partId = k;
            energy_find_neighborhood(points, partId, p->cutoff, nindices, bindices);
            etot[i] = energy_minimize_neighborhood(p, nindices, bindices, points, forces);
        }
        i = (i + 1) % 3;
        ntot += 1;
        de[0] = etot[0] - etot[1];
        de[1] = etot[1] - etot[2];
        de[2] = etot[2] - etot[0];
        de_avg = (de[0]*de[0] + de[1]*de[1] + de[2]*de[2])/3;
        std::cout << "n:" << ntot << ", de:" << de_avg << std::endl;
    }
    while(ntot < 3 && (ntot < p->max_outer_iter));
}


float sphere_2body(EnergyMinimizer* p, Magnum::Vector3 *x1, Magnum::Vector3 *x2,
                   Magnum::Vector3 *f1, Magnum::Vector3 *f2) {
    Magnum::Vector3 dx = (*x2 - *x1); // vector from x1 -> x2
    float r = dx.length() + 0.01; // softness factor.
    if(r > p->cutoff) {
        return 0;
    }
    
    float f = 0.0001 / (r * r);
    
    *f1 = *f1 - f * dx / (2 * r);
    
    if(f2) {
        *f2 = *f2 + f * dx / (2 * r);
    }
    
    return std::abs(f);
}

float sphere_1body(EnergyMinimizer* p, Magnum::Vector3 *p1,
                   Magnum::Vector3 *f1) {
    float r = (*p1).length();

    float f = 1 * (1.0 - r); // magnitude of force.
    
    *f1 = *f1 + (f/r) * (*p1);
    
    return std::abs(f);
}


// Yes, Windows has the __cpuid and __cpuidx macros in the #include <intrin.h>
// header file, but it seg-faults when we try to call them from clang.
// this version of the cpuid seems to work with clang on both Windows and mac.

// adapted from https://github.com/01org/linux-sgx/blob/master/common/inc/internal/linux/cpuid_gnu.h
/* This is a PIC-compliant version of CPUID */
static inline void __mx_cpuid(int *eax, int *ebx, int *ecx, int *edx)
{
#if defined(__x86_64__)
    asm("cpuid"
            : "=a" (*eax),
            "=b" (*ebx),
            "=c" (*ecx),
            "=d" (*edx)
            : "0" (*eax), "2" (*ecx));

#else
    /*on 32bit, ebx can NOT be used as PIC code*/
    asm volatile ("xchgl %%ebx, %1; cpuid; xchgl %%ebx, %1"
            : "=a" (*eax), "=r" (*ebx), "=c" (*ecx), "=d" (*edx)
            : "0" (*eax), "2" (*ecx));
#endif
}

#ifdef _WIN32

// TODO: PATHETIC HACK for windows. 
// don't know why, but calling cpuid in release mode, and ONLY in release 
// mode causes a segfault. Hack is to flush stdout, push some junk on the stack. 
// and force a task switch. 
// dont know why this works, but if any of these are not here, then it segfaults
// in release mode. 
// this also seems to work, but force it non-inline and return random
// number. 
// Maybe the optimizer is inlining it, and inlining causes issues
// calling cpuid??? 

static __declspec(noinline) int mx_cpuid(int a[4], int b)
{
    a[0] = b;
    a[2] = 0;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
    return std::rand();
}

static __declspec(noinline) int mx_cpuidex(int a[4], int b, int c)
{
    a[0] = b;
    a[2] = c;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
    return std::rand();
}

#else 

static  void mx_cpuid(int a[4], int b)
{
    a[0] = b;
    a[2] = 0;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
}

static void mx_cpuidex(int a[4], int b, int c)
{
    a[0] = b;
    a[2] = c;
    __mx_cpuid(&a[0], &a[1], &a[2], &a[3]);
}

#endif

             
// InstructionSet.cpp
// Compile by using: cl /EHsc /W4 InstructionSet.cpp
// processor: x86, x64
// Uses the __cpuid intrinsic to get information about
// CPU extended instruction set support.



typedef Magnum::Vector4i VectorType;


class InstructionSet
{

private:
    

    class InstructionSet_Internal
    {
    public:
        InstructionSet_Internal() :
                nIds_ { 0 }, nExIds_ { 0 }, isIntel_ { false }, isAMD_ { false }, f_1_ECX_ {
                        0 }, f_1_EDX_ { 0 }, f_7_EBX_ { 0 }, f_7_ECX_ { 0 }, f_81_ECX_ {
                        0 }, f_81_EDX_ { 0 }, data_ { }, extdata_ { }
        {
            VectorType cpui;

            // Calling mx_cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            mx_cpuid(cpui.data(), 0);
            nIds_ = cpui[0];

            for (int i = 0; i <= nIds_; ++i) {
                mx_cpuidex(cpui.data(), i, 0);
                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel") {
                isIntel_ = true;
            } else if (vendor_ == "AuthenticAMD") {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1) {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7) {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling mx_cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            mx_cpuid(cpui.data(), 0x80000000);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= nExIds_; ++i) {
                mx_cpuidex(cpui.data(), i, 0);
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001) {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004) {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };


        int nIds_;
        int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<VectorType> data_;
        std::vector<VectorType> extdata_;
    };

    const InstructionSet_Internal CPU_Rep;


public:
    // getters
    std::string Vendor(void)
    {
        return CPU_Rep.vendor_;
    }
    std::string Brand(void)
    {
        return CPU_Rep.brand_;
    }

    bool SSE3(void)
    {
        return CPU_Rep.f_1_ECX_[0];
    }
    bool PCLMULQDQ(void)
    {
        return CPU_Rep.f_1_ECX_[1];
    }
    bool MONITOR(void)
    {
        return CPU_Rep.f_1_ECX_[3];
    }
    bool SSSE3(void)
    {
        return CPU_Rep.f_1_ECX_[9];
    }
    bool FMA(void)
    {
        return CPU_Rep.f_1_ECX_[12];
    }
    bool CMPXCHG16B(void)
    {
        return CPU_Rep.f_1_ECX_[13];
    }
    bool SSE41(void)
    {
        return CPU_Rep.f_1_ECX_[19];
    }
    bool SSE42(void)
    {
        return CPU_Rep.f_1_ECX_[20];
    }
    bool MOVBE(void)
    {
        return CPU_Rep.f_1_ECX_[22];
    }
    bool POPCNT(void)
    {
        return CPU_Rep.f_1_ECX_[23];
    }
    bool AES(void)
    {
        return CPU_Rep.f_1_ECX_[25];
    }
    bool XSAVE(void)
    {
        return CPU_Rep.f_1_ECX_[26];
    }
    bool OSXSAVE(void)
    {
        return CPU_Rep.f_1_ECX_[27];
    }
    bool AVX(void)
    {
        return CPU_Rep.f_1_ECX_[28];
    }
    bool F16C(void)
    {
        return CPU_Rep.f_1_ECX_[29];
    }
    bool RDRAND(void)
    {
        return CPU_Rep.f_1_ECX_[30];
    }

    bool MSR(void)
    {
        return CPU_Rep.f_1_EDX_[5];
    }
    bool CX8(void)
    {
        return CPU_Rep.f_1_EDX_[8];
    }
    bool SEP(void)
    {
        return CPU_Rep.f_1_EDX_[11];
    }
    bool CMOV(void)
    {
        return CPU_Rep.f_1_EDX_[15];
    }
    bool CLFSH(void)
    {
        return CPU_Rep.f_1_EDX_[19];
    }
    bool MMX(void)
    {
        return CPU_Rep.f_1_EDX_[23];
    }
    bool FXSR(void)
    {
        return CPU_Rep.f_1_EDX_[24];
    }
    bool SSE(void)
    {
        return CPU_Rep.f_1_EDX_[25];
    }
    bool SSE2(void)
    {
        return CPU_Rep.f_1_EDX_[26];
    }

    bool FSGSBASE(void)
    {
        return CPU_Rep.f_7_EBX_[0];
    }
    bool BMI1(void)
    {
        return CPU_Rep.f_7_EBX_[3];
    }
    bool HLE(void)
    {
        return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4];
    }
    bool AVX2(void)
    {
        return CPU_Rep.f_7_EBX_[5];
    }
    bool BMI2(void)
    {
        return CPU_Rep.f_7_EBX_[8];
    }
    bool ERMS(void)
    {
        return CPU_Rep.f_7_EBX_[9];
    }
    bool INVPCID(void)
    {
        return CPU_Rep.f_7_EBX_[10];
    }
    bool RTM(void)
    {
        return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11];
    }
    bool AVX512F(void)
    {
        return CPU_Rep.f_7_EBX_[16];
    }
    bool RDSEED(void)
    {
        return CPU_Rep.f_7_EBX_[18];
    }
    bool ADX(void)
    {
        return CPU_Rep.f_7_EBX_[19];
    }
    bool AVX512PF(void)
    {
        return CPU_Rep.f_7_EBX_[26];
    }
    bool AVX512ER(void)
    {
        return CPU_Rep.f_7_EBX_[27];
    }
    bool AVX512CD(void)
    {
        return CPU_Rep.f_7_EBX_[28];
    }
    bool SHA(void)
    {
        return CPU_Rep.f_7_EBX_[29];
    }

    bool PREFETCHWT1(void)
    {
        return CPU_Rep.f_7_ECX_[0];
    }

    bool LAHF(void)
    {
        return CPU_Rep.f_81_ECX_[0];
    }
    bool LZCNT(void)
    {
        return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5];
    }
    bool ABM(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5];
    }
    bool SSE4a(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6];
    }
    bool XOP(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11];
    }
    bool TBM(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21];
    }

    bool SYSCALL(void)
    {
        return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11];
    }
    bool MMXEXT(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22];
    }
    bool RDTSCP(void)
    {
        return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27];
    }
    bool _3DNOWEXT(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30];
    }
    bool _3DNOW(void)
    {
        return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31];
    }
};

// Initialize static member data
//const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

PyObject *MxCompileFlagsDict(PyObject *o) {
    PyObject *dict = PyDict_New();
    
    // Defining Lambda function and
    // Capturing Local variables by Value
    auto add_item = [dict] (const char* key, bool value) {
        PyDict_SetItemString(dict, key, PyBool_FromLong(value));
    };

#ifdef _DEBUG
    add_item("_DEBUG", true);
#else
    add_item("_DEBUG", 0);
#endif 

    add_item("MX_OPENMP", MX_OPENMP);
    add_item("MX_OPENMP_BONDS", MX_OPENMP_BONDS);
    add_item("MX_OPENMP_INTEGRATOR", MX_OPENMP_INTEGRATOR);
    add_item("MX_VECTORIZE_FLUX", MX_VECTORIZE_FLUX);
    add_item("MX_VECTORIZE_FORCE", MX_VECTORIZE_FORCE);
    add_item("MX_SSE42", MX_SSE42);
    add_item("MX_AVX", MX_AVX);
    add_item("MX_AVX2", MX_AVX2);
    
#ifdef MX_THREADING
    add_item("MX_THREADING", true);
    PyDict_SetItemString(dict, "MX_THREADPOOL_SIZE", PyLong_FromLong(mx::ThreadPool::hardwareThreadSize()));
#else
    add_item("MX_THREADING", false);
    PyDict_SetItemString(dict, "MX_THREADPOOL_SIZE", PyLong_FromLong(0));
#endif
    
    PyDict_SetItemString(dict, "MX_SIMD_SIZE", PyLong_FromLong(MX_SIMD_SIZE));

#ifdef __SSE__
    add_item("__SSE__", __SSE__);
#else
    add_item("__SSE__", 0);
#endif
    
#ifdef __SSE2__
    add_item("__SSE2__", __SSE2__);
#else
    add_item("__SSE2__", 0);
#endif
    
#ifdef __SSE3__
    add_item("__SSE3__", __SSE3__);
#else
    add_item("__SSE3__", 0);
#endif
    
#ifdef __SSE4_2__
    add_item("__SSE4_2__", __SSE4_2__);
#else
    add_item("__SSE4_2__", 0);
#endif
    
#ifdef __AVX__
    add_item("__AVX__", __AVX__);
#else
    add_item("__AVX__", 0);
#endif
    
#ifdef __AVX2__
    add_item("__AVX2__", __AVX2__);
#else
    add_item("__AVX2__", 0);
#endif
    
    return dict;
}
          
PyObject *MxInstructionSetFeatruesDict(PyObject *o) {
     
     PyObject *dict = PyDict_New();

     InstructionSet is;
     
     // Defining Lambda function and
     // Capturing Local variables by Value
     auto add_item = [dict] (const char* key, bool value) {
         PyDict_SetItemString(dict, key, PyBool_FromLong(value));
     };
     
     PyObject *s = carbon::cast(is.Vendor());
     PyDict_SetItemString(dict, "VENDOR", s);
     Py_DecRef(s);
     
     s = carbon::cast(is.Brand());
     PyDict_SetItemString(dict, "ID", s);
     Py_DecRef(s);
     
     add_item("3DNOW", is._3DNOW());
     add_item("3DNOWEXT", is._3DNOWEXT());
     add_item("ABM", is.ABM());
     add_item("ADX", is.ADX());
     add_item("AES", is.AES());
     add_item("AVX", is.AVX());
     add_item("AVX2", is.AVX2());
     add_item("AVX512CD", is.AVX512CD());
     add_item("AVX512ER", is.AVX512ER());
     add_item("AVX512F", is.AVX512F());
     add_item("AVX512PF", is.AVX512PF());
     add_item("BMI1", is.BMI1());
     add_item("BMI2", is.BMI2());
     add_item("CLFSH", is.CLFSH());
     add_item("CMPXCHG16B", is.CMPXCHG16B());
     add_item("CX8", is.CX8());
     add_item("ERMS", is.ERMS());
     add_item("F16C", is.F16C());
     add_item("FMA", is.FMA());
     add_item("FSGSBASE", is.FSGSBASE());
     add_item("FXSR", is.FXSR());
     add_item("HLE", is.HLE());
     add_item("INVPCID", is.INVPCID());
     add_item("LAHF", is.LAHF());
     add_item("LZCNT", is.LZCNT());
     add_item("MMX", is.MMX());
     add_item("MMXEXT", is.MMXEXT());
     add_item("MONITOR", is.MONITOR());
     add_item("MOVBE", is.MOVBE());
     add_item("MSR", is.MSR());
     add_item("OSXSAVE", is.OSXSAVE());
     add_item("PCLMULQDQ", is.PCLMULQDQ());
     add_item("POPCNT", is.POPCNT());
     add_item("PREFETCHWT1", is.PREFETCHWT1());
     add_item("RDRAND", is.RDRAND());
     add_item("RDSEED", is.RDSEED());
     add_item("RDTSCP", is.RDTSCP());
     add_item("RTM", is.RTM());
     add_item("SEP", is.SEP());
     add_item("SHA", is.SHA());
     add_item("SSE", is.SSE());
     add_item("SSE2", is.SSE2());
     add_item("SSE3", is.SSE3());
     add_item("SSE4.1", is.SSE41());
     add_item("SSE4.2", is.SSE42());
     add_item("SSE4a", is.SSE4a());
     add_item("SSSE3", is.SSSE3());
     add_item("SYSCALL", is.SYSCALL());
     add_item("TBM", is.TBM());
     add_item("XOP", is.XOP());
     add_item("XSAVE", is.XSAVE());
     
     return dict;
 }
 
 uint64_t MxInstructionSetFeatures() {

     InstructionSet is;
     
     uint64_t result = 0;
     
     auto add_item = [&result] (uint64_t key, bool value) {
         if(value) {
             result |= key;
         }
     };
     
     add_item(IS_3DNOW, is._3DNOW());
     add_item(IS_3DNOWEXT, is._3DNOWEXT());
     add_item(IS_ABM, is.ABM());
     add_item(IS_ADX, is.ADX());
     add_item(IS_AES, is.AES());
     add_item(IS_AVX, is.AVX());
     add_item(IS_AVX2, is.AVX2());
     add_item(IS_AVX512CD, is.AVX512CD());
     add_item(IS_AVX512ER, is.AVX512ER());
     add_item(IS_AVX512F, is.AVX512F());
     add_item(IS_AVX512PF, is.AVX512PF());
     add_item(IS_BMI1, is.BMI1());
     add_item(IS_BMI2, is.BMI2());
     add_item(IS_CLFSH, is.CLFSH());
     add_item(IS_CMPXCHG16B, is.CMPXCHG16B());
     add_item(IS_CX8, is.CX8());
     add_item(IS_ERMS, is.ERMS());
     add_item(IS_F16C, is.F16C());
     add_item(IS_FMA, is.FMA());
     add_item(IS_FSGSBASE, is.FSGSBASE());
     add_item(IS_FXSR, is.FXSR());
     add_item(IS_HLE, is.HLE());
     add_item(IS_INVPCID, is.INVPCID());
     add_item(IS_LAHF, is.LAHF());
     add_item(IS_LZCNT, is.LZCNT());
     add_item(IS_MMX, is.MMX());
     add_item(IS_MMXEXT, is.MMXEXT());
     add_item(IS_MONITOR, is.MONITOR());
     add_item(IS_MOVBE, is.MOVBE());
     add_item(IS_MSR, is.MSR());
     add_item(IS_OSXSAVE, is.OSXSAVE());
     add_item(IS_PCLMULQDQ, is.PCLMULQDQ());
     add_item(IS_POPCNT, is.POPCNT());
     add_item(IS_PREFETCHWT1, is.PREFETCHWT1());
     add_item(IS_RDRAND, is.RDRAND());
     add_item(IS_RDSEED, is.RDSEED());
     add_item(IS_RDTSCP, is.RDTSCP());
     add_item(IS_RTM, is.RTM());
     add_item(IS_SEP, is.SEP());
     add_item(IS_SHA, is.SHA());
     add_item(IS_SSE, is.SSE());
     add_item(IS_SSE2, is.SSE2());
     add_item(IS_SSE3, is.SSE3());
     add_item(IS_SSE41, is.SSE41());
     add_item(IS_SSE42, is.SSE42());
     add_item(IS_SSE4a, is.SSE4a());
     add_item(IS_SSSE3, is.SSSE3());
     add_item(IS_SYSCALL, is.SYSCALL());
     add_item(IS_TBM, is.TBM());
     add_item(IS_XOP, is.XOP());
     add_item(IS_XSAVE, is.XSAVE());
     
     return result;
 }

             

//  Windows
#ifdef _WIN32
#include <Windows.h>
double MxWallTime() {
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double MxCPUTime(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
        (double)(d.dwLowDateTime |
                 ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double MxWallTime() {
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double MxCPUTime(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif




WallTime::WallTime() {
    start = MxWallTime();
}

WallTime::~WallTime() {
    _Engine.wall_time += (MxWallTime() - start);
}
   

