/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FileImage.h
*
******************************************************************/

#ifndef _CX3D_FILEIMAGE_H_
#define _CX3D_FILEIMAGE_H_

#include <x3d/CyberX3DConfig.h>

#ifdef CX3D_SUPPORT_OLDCPP
#include <OldCpp.h>
#endif

#include <x3d/FileUtil.h>

namespace CyberX3D {

#if !R && !G && !B
#define R	0
#define G	1
#define B	2
#endif

typedef unsigned char RGBColor24[3];
typedef unsigned char RGBAColor32[4];

class FileImage {

public:

	FileImage();
	virtual ~FileImage();

	bool isOk() const;
	
	virtual int			getFileType() const = 0;

	virtual int			getWidth() const = 0;
	virtual int			getHeight() const = 0;
	virtual RGBColor24	*getImage() const = 0;

	virtual bool hasTransparencyColor() const {
		return false;
	}

	virtual void getTransparencyColor(RGBColor24 color) const {
	};

	RGBColor24	*getImage(int newx, int newy) const;
	RGBAColor32	*getRGBAImage() const;
	RGBAColor32	*getRGBAImage(int newx, int newy) const;
};

}

#endif
