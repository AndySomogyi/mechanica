/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: Graphic3D.h
*
******************************************************************/

#ifndef _CX3D_GRAPHIC3D_H_
#define _CX3D_GRAPHIC3D_H_

#include <x3d/CyberX3DConfig.h>

#ifdef CX3D_SUPPORT_OPENGL

#ifdef WIN32
#include <windows.h>
#endif // WIN32

#ifdef MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#endif // MACOSX

#ifdef WIN32
typedef GLvoid (CALLBACK*GLUtessCallBackFunc)(void);
#else
typedef GLvoid (*GLUtessCallBackFunc)(void);
#endif // WIN32

#endif // CX3D_SUPPORT_OPENGL

#endif
