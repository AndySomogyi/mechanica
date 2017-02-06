/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FileJPEG.h
*
******************************************************************/

#ifndef _CX3D_FILEJPEG_H_
#define _CX3D_FILEJPEG_H_

#include <x3d/FileImage.h>

#ifdef CX3D_SUPPORT_JPEG

namespace CyberX3D {

class FileJPEG : public FileImage {
	int			width;
	int			height;
	RGBColor24	*imgBuffer;

public:	

	FileJPEG(const char *filename);
	virtual ~FileJPEG();
	
	bool load(const char *filename);

	int getFileType() const {
		return FILE_FORMAT_JPEG;
	}

	int getWidth() const {
		return width;
	}

	int getHeight() const {
		return height;
	}
	
	RGBColor24 *getImage() const {
		return imgBuffer;
	}
};

}

#endif

#endif
