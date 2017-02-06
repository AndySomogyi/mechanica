/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MultiTextureCoordinateNode.h
*
******************************************************************/

#ifndef _CX3D_MULTITEXTURECOORDINATENODE_H_
#define _CX3D_MULTITEXTURECOORDINATENODE_H_

#include <x3d/X3DFields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class MultiTextureCoordinateNode : public Node
{
	MFNode *texCoordField;

public:

	MultiTextureCoordinateNode();
	virtual ~MultiTextureCoordinateNode();

	////////////////////////////////////////////////
	//	texCoord
	////////////////////////////////////////////////
	
	MFNode *getTexCoordField() const;

	void addTexCoord(Node *value);
	int getNTexCoords() const;
	Node *getTexCoord(int index) const;

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

	MultiTextureCoordinateNode *next() const;
	MultiTextureCoordinateNode *nextTraversal() const;
};

}

#endif
