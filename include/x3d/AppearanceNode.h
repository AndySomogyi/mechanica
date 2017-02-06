/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	AppearanceNode.h
*
******************************************************************/

#ifndef _CX3D_APPEARANCENODE_H_
#define _CX3D_APPEARANCENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>
#include <x3d/MaterialNode.h>
#include <x3d/TextureTransformNode.h>

namespace CyberX3D {

class AppearanceNode : public Node {

	SFNode *materialField;
	SFNode *textureField;
	SFNode *texTransformField;
	SFNode *linePropertiesField;
	SFNode *fillPropertiesField;

public:

	AppearanceNode();
	virtual ~AppearanceNode();

	////////////////////////////////////////////////
	//	SFNode Field
	////////////////////////////////////////////////

	SFNode *getMaterialField() const;
	SFNode *getTextureField() const ;
	SFNode *getTextureTransformField() const;
	SFNode *getLinePropertiesField() const;
	SFNode *getFillPropertiesField() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	AppearanceNode *next() const;
	AppearanceNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	virtual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();
	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
