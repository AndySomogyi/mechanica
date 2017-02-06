/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FilePNG.h
*
******************************************************************/

#ifndef _CX3D_FILEPNG_H_
#define _CX3D_FILEPNG_H_

#include <x3d/FileImage.h>

#ifdef CX3D_SUPPORT_PNG

namespace CyberX3D {

class FilePNG : public FileImage {
	int			mWidth;
	int			mHeight;
	RGBColor24	*mImgBuffer;
	bool		mHasTransparencyColor;
	RGBColor24	mTransparencyColor;
public:	

	FilePNG(const char *filename);
	virtual ~FilePNG();
	
	bool load(const char *filename);

	int getFileType() const {
		return FILE_FORMAT_PNG;
	}

	int getWidth() const {
		return mWidth;
	}

	int getHeight() const {
		return mHeight;
	}
	
	RGBColor24 *getImage() const {
		return mImgBuffer;
	}

	bool hasTransparencyColor() const {
		return mHasTransparencyColor;
	}

	void setTransparencyColor(RGBColor24 color);

	void getTransparencyColor(RGBColor24 color) const;

};

}

#endif

#endif
