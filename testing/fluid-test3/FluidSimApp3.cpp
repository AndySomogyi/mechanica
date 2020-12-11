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


#include <Mechanica.h>
#include <MxSimulator.h>


#ifdef _WIN32
INT WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    PSTR lpCmdLine, INT nCmdShow) {
#else
int main(int argc, char** argv) {
    C_UNSUSED(argc)
    C_UNSUSED(argv)
#endif

    Mx_Initialize(0);
    MxSimulator::Config conf;
    MxSimulator::GLConfig glConf;

    conf.example = "argon";
    MxSimulator_InitConfig(conf, glConf);

    MxSimulator_Run();

    return 0;
}





