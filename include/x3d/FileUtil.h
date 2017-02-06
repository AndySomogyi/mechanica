/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FileUtil.h
*
******************************************************************/

#ifndef _CX3D_FILEUTIL_H_
#define _CX3D_FILEUTIL_H_

#include <x3d/CyberX3DConfig.h>

namespace CyberX3D {

enum {
FILE_FORMAT_NONE,
FILE_FORMAT_VRML,
FILE_FORMAT_XML,
FILE_FORMAT_X3D,
FILE_FORMAT_GIF,
FILE_FORMAT_JPEG,
FILE_FORMAT_TARGA,
FILE_FORMAT_PNG
};

int GetFileFormat(const char *filename);

}

#endif
