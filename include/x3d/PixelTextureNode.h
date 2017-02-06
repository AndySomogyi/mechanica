/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PixelTextureNode.h
*
******************************************************************/

#ifndef _CX3D_PIXELTEXTURENODE_H_
#define _CX3D_PIXELTEXTURENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Texture2DNode.h>

namespace CyberX3D {

class PixelTextureNode : public Texture2DNode {

	SFImage *imageField;
	
public:

	PixelTextureNode();
	virtual ~PixelTextureNode();

	////////////////////////////////////////////////
	// Image
	////////////////////////////////////////////////

	SFImage *getImageField() const;

	void addImage(int value);
	int getNImages() const;
	int getImage(int index) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	PixelTextureNode *next() const;
	PixelTextureNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Imagemation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

