/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FileTarga.h
*
******************************************************************/

#ifndef _CX3D_FILETARGA_H_
#define _CX3D_FILETARGA_H_

#include <x3d/FileImage.h>

namespace CyberX3D {

typedef struct {
	unsigned char		IDLength;
	unsigned char		CoMapType;
	unsigned char		ImgType;
	unsigned short int	Index;	
	unsigned short int	Length;	
	unsigned char		CoSize;	
	unsigned short int	XOrg;	
	unsigned short int	YOrg;	
	unsigned short int	Width;	
	unsigned short int	Height;	
	unsigned char		PixelSize;
	unsigned char		AttBits;
} TargaHeadeInfor;

class FileTarga : public FileImage {
	unsigned char		idLength;
	unsigned char		coMapType;
	unsigned char		imgType;
	unsigned short int	index;	
	unsigned short int	length;	
	unsigned char		coSize;	
	unsigned short int	xOrg;	
	unsigned short int	yOrg;	
	unsigned short int	width;	
	unsigned short int	height;	
	unsigned char		pixelSize;
	unsigned char		attBits;
	RGBColor24			*imageBuffer;
public:
	FileTarga(const char *filename);
	FileTarga(int cx, int cy, RGBColor24 *color);
	virtual ~FileTarga();

	void		initialize();
	bool		load(const char *filename);
	bool		save(const char *filename) const;

	int			getFileType() const { return FILE_FORMAT_TARGA; }

	int			getWidth() const { return width; }
	int			getHeight() const { return height; }
	RGBColor24	*getImage()	const { return imageBuffer; }
};

}

#endif
