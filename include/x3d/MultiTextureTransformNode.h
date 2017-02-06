/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MultiTextureTransformNode.h
*
******************************************************************/

#ifndef _CX3D_MULTITEXTURETRANSFORMNODE_H_
#define _CX3D_MULTITEXTURETRANSFORMNODE_H_

#include <x3d/X3DFields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class MultiTextureTransformNode : public Node
{
	MFNode *textureTransformField;

public:

	MultiTextureTransformNode();
	virtual ~MultiTextureTransformNode();

	////////////////////////////////////////////////
	//	textureTransform
	////////////////////////////////////////////////
	
	MFNode *getTextureTransformField() const;

	void addTextureTransform(Node *value);
	int getNTextureTransforms() const;
	Node *getTextureTransform(int index) const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	MultiTextureTransformNode *next() const;
	MultiTextureTransformNode *nextTraversal() const;
};

}

#endif
