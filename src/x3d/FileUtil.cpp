/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FileUtil.cpp
*
******************************************************************/

#include <stdio.h>
#include <string.h>
#include <x3d/FileUtil.h>
#if defined(CX3D_SUPPORT_GZIP)
#include <zlib.h>
#endif

using namespace CyberX3D;

////////////////////////////////////////////////
//	GetFileFormat
////////////////////////////////////////////////

int CyberX3D::GetFileFormat(const char *filename) 
{
	unsigned char signature[5];

#if defined(CX3D_SUPPORT_GZIP)
	gzFile fp = gzopen(filename, "rb");
	if (!fp)	
		return FILE_FORMAT_NONE;
	if (gzread(fp, signature, 5 * sizeof(unsigned char)) != 5) {
		gzclose(fp);
		return FILE_FORMAT_NONE;
	}
	gzclose(fp);
#else
	FILE *fp = fopen(filename, "rb");
	if (!fp)	
		return FILE_FORMAT_NONE;
	if (fread(signature, sizeof(unsigned char), 5, fp) != 5) {
		fclose(fp);
		return FILE_FORMAT_NONE;
	}
	fclose(fp);
#endif

	int fileType = FILE_FORMAT_NONE;

	if (!strncmp("#VRML", (char *)signature, 5))
		fileType = FILE_FORMAT_VRML;

	if (!strncmp("<?xml", (char *)signature, 5))
		fileType = FILE_FORMAT_XML;

	if (!strncmp("GIF", (char *)signature, 3))
		fileType = FILE_FORMAT_GIF;

	if (signature[0] == 0xff && signature[1] == 0xd8)
		fileType = FILE_FORMAT_JPEG;

	if (!strncmp("PNG", (char *)(signature+1), 3))
		fileType = FILE_FORMAT_PNG;

	return fileType;
}
