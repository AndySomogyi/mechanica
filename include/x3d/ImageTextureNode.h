/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ImageTextureNode.h
*
******************************************************************/

#ifndef _CX3D_IMAGETEXTURENODE_H_
#define _CX3D_IMAGETEXTURENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Texture2DNode.h>
#include <x3d/FileImage.h>

namespace CyberX3D {

class ImageTextureNode : public Texture2DNode {

	MFString *urlField;

	int			mWidth;
	int			mHeight;
	FileImage	*mFileImage;
	RGBAColor32	*mImageBuffer;
	String		mCurrentTextureName;

public:

	ImageTextureNode();
	virtual ~ImageTextureNode();

	////////////////////////////////////////////////
	// Url
	////////////////////////////////////////////////

	MFString *getUrlField() const;

	void addUrl(const char * value);
	int getNUrls() const;
	const char *getUrl(int index) const;
	void setUrl(int index, const char *urlString);

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ImageTextureNode *next() const 
	{
		return (ImageTextureNode *)Node::next(getType());
	}

	ImageTextureNode *nextTraversal() const 
	{
		return (ImageTextureNode *)Node::nextTraversalByType(getType());
	}

	////////////////////////////////////////////////
	//	Image
	////////////////////////////////////////////////

	bool createImage();
	
	int getWidth() const
	{
		return mWidth;
	}
	
	int getHeight() const
	{
		return mHeight;
	}
	
	RGBAColor32 *getImage() const
	{
		return mImageBuffer;
	}
	
	FileImage *getFileImage() const
	{
		return mFileImage;
	}

	////////////////////////////////////////////////
	//	virtual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const
	{
		return false;
	}

	void initialize();

	void uninitialize();

	void setCurrentTextureName(const char *name) 
	{
		mCurrentTextureName.setValue(name);
	}

	const char *getCurrentTextureName() const
	{
		return mCurrentTextureName.getValue();
	}

	void updateTexture();

	void update();

	////////////////////////////////////////////////
	//	infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
