/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GIF89a.h
*
******************************************************************/

#ifndef _CX3D_GIF89A_H_
#define _CX3D_GIF89A_H_

#include <x3d/FileImage.h>

namespace CyberX3D {

typedef struct {
	unsigned char	signature[3];
	unsigned char	version[3];
	unsigned short	width;
	unsigned short	height;
	unsigned char	packedField;
	unsigned char	bgColorIndex;
	unsigned char	aspectRaito;
} GIF89aHeaderInfo;

typedef struct {
	unsigned short	imageLeftPosition;
	unsigned short	imageTopPosition;
	unsigned short	imageWidth;
	unsigned short	imageHeight;
	unsigned char	packedField;
} GIF89aImageInfo;

typedef struct {
	unsigned int	n;
	unsigned int	*data;
} GIF89aLzwTable;

typedef struct {
	GIF89aImageInfo		info;
	RGBColor24			*localColorTable;
	RGBColor24			*buffer;
	unsigned int		bufferSize;
	bool				transparencyFlag;
	unsigned int		transparencyColorIndex;
} GIF89aImage;

#if !R && !G && !B
#define R	0
#define G	1
#define B	2
#endif

#define GIF89A_LZW_TABLE_SIZE	4096

class FileGIF89a : public FileImage {
	GIF89aHeaderInfo	headerInfo;
	RGBColor24			*globalColorTable;

	int					nImage;
	GIF89aImage			*image;

	GIF89aLzwTable		lzwTable[GIF89A_LZW_TABLE_SIZE];
	unsigned int		lzwCodeSize;
	unsigned int		lzwClearCode;
	unsigned int		lzwEndCode;
	unsigned int		lzwOffsetBit;

	unsigned char		*lzwBuffer;
	unsigned int		lzwBufferOffset;
	unsigned int		lzwBufferSize;
	unsigned int		lzwTableIndex;
	
public:

	FileGIF89a(const char *fname);
	virtual ~FileGIF89a();

	bool	load(const char *fname);

	/////////////////////////////////////
	//	Header Infomation
	/////////////////////////////////////

	unsigned int getGlobalWidth() const {
		return (unsigned int)headerInfo.width;
	}

	unsigned int getGlobalHeight() const {
		return (unsigned int)headerInfo.height;
	}

	unsigned int getGlobalColorTableFlag() const {
		return (unsigned int)((headerInfo.packedField & 0x80) >> 7);
	}

	unsigned int getColorResolution() const {
		return (unsigned int)((headerInfo.packedField & 0x70) >> 4);
	}

	unsigned int getSortFlag() const {
		return (unsigned int)((headerInfo.packedField & 0x08) >> 3);
	}

	unsigned int getSizeOfGlobalTable() const {
		return (unsigned int)(headerInfo.packedField & 0x07);
	}

	unsigned int getBgColorIndex() const {
		return (unsigned int)headerInfo.bgColorIndex;
	}

	unsigned int getAspectRaito() const {
		return (unsigned int)headerInfo.aspectRaito;
	}

	/////////////////////////////////////
	//	Image Infomation
	/////////////////////////////////////

	int getNImages() const {
		return nImage;
	}

	GIF89aImageInfo	*getImageInfo(int n) const {
		return &image[n].info;
	}

	RGBColor24 *getImageBuffer(int n) const { 
		return image[n].buffer; 
	}

	int getImageBufferSize(int n) const { 
		return image[n].bufferSize; 
	}

	unsigned int getImageLeftPosition(int n)	const {
		return (unsigned int)image[n].info.imageLeftPosition;
	}

	unsigned int getImageTopPosition(int n)	const {
		return (unsigned int)image[n].info.imageTopPosition;
	}

	unsigned int getImageWidth(int n)	const {
		return (unsigned int)image[n].info.imageWidth;
	}

	unsigned int getImageHeight(int n)	const {
		return (unsigned int)image[n].info.imageHeight;
	}

	unsigned int getImageLocalColorTableFlag(int n) const {
		return (unsigned int)((image[n].info.packedField & 0x80) >> 7);
	}

	unsigned int getImageInterlaceFlag(int n) const {
		return (unsigned int)((image[n].info.packedField & 0x40) >> 6);
	}

	unsigned int getImageSortFlag(int n) const {
		return (unsigned int)((image[n].info.packedField & 0x20) >> 5);
	}

	unsigned int getImageSizeOfLocalTable(int n) const {
		return (unsigned int)(image[n].info.packedField & 0x07);
	}

	bool hasTransparencyColor(int n) const {
		return image[n].transparencyFlag;
	}

	unsigned int getTransparencyColorIndex(int n) const {
		return image[n].transparencyColorIndex;
	}

	void convertInterlacedImage(int n);

	/////////////////////////////////////
	//	LZW Decompress
	/////////////////////////////////////

	void initializeLzwTable(unsigned int codeSize, unsigned char *dataByte, unsigned int dataSize);
	void reinitializeLzwTable();
	void terminateLzwTable();

	void initializeImageBuffer(int n);

	unsigned int getNextCode(unsigned int bitSize);

	void getColor(int n, unsigned int index, RGBColor24 color) const;

	void setLzwCodeSize(unsigned int codeSize) {
		lzwCodeSize = codeSize;
	}

	unsigned int getLzwCodeSize() const {
		return lzwCodeSize;
	}

	void setClearCode(unsigned int code) {
		lzwClearCode = code;
	}

	unsigned int getClearCode() const {
		return lzwClearCode;
	}

	void setEndCode(unsigned int code) {
		lzwEndCode = code;
	}

	unsigned int getEndCode() const {
		return lzwEndCode;
	}

	void setLzwBuffer(unsigned char *buffer) {
		lzwBuffer = buffer;
	}

	unsigned char *getLzwBuffer() const {
		return lzwBuffer;
	}

	void setLzwBufferSize(unsigned int size) {
		lzwBufferSize = size;
	}

	unsigned int getLzwBufferSize() const {
		return lzwBufferSize;
	}

	void setLzwBufferOffset(unsigned int offset) {
		lzwBufferOffset = offset;
	}

	unsigned int getLzwBufferOffset() const {
		return lzwBufferOffset;
	}

	void setLzwTableIndex(unsigned int index) {
		lzwTableIndex = index;
	}

	unsigned int getLzwTableIndex() const {
		return lzwTableIndex;
	}

	void			setLzwTable(unsigned int index, unsigned int value);
	unsigned int	addLzwTable(unsigned int sindex, unsigned int cindex, unsigned int codeBitSize);

	void			outputData(int n, unsigned int tableIndex) const;
	void			outputFirstData(int n, unsigned int tableIndex) const;
	void			outputData(int n, unsigned int sindex, unsigned int cindex) const;

	/////////////////////////////////////
	//	Virtual function
	/////////////////////////////////////

	int getFileType() const {
		return FILE_FORMAT_GIF;
	}

	int getWidth() const {
		return (0 < nImage) ? getImageWidth(0) : 0;
	}

	int getHeight() const {
		return (0 < nImage) ? getImageHeight(0) : 0;
	}
	
	RGBColor24 *getImage() const {
		return (0 < nImage) ? getImageBuffer(0) : 0;
	}

	bool hasTransparencyColor() const {
		return hasTransparencyColor(0);
	}

	void getTransparencyColor(RGBColor24 color) const {
		if (hasTransparencyColor(0))
			getColor(0, getTransparencyColorIndex(0), color);
	}

	/////////////////////////////////////
	//	Output infomations
	/////////////////////////////////////
	
	void	printHeaderInfo() const;
	void	printImageInfo(int n) const;
};

}

#endif
