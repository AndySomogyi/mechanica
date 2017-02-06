/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TextureCoordinateNode.h
*
******************************************************************/

#ifndef _CX3D_TEXTURECOORDINATENODE_H_
#define _CX3D_TEXTURECOORDINATENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/GeometricPropertyNode.h>

namespace CyberX3D {

class TextureCoordinateNode : public GeometricPropertyNode 
{
	
	MFVec2f *pointField;

public:

	TextureCoordinateNode();
	virtual ~TextureCoordinateNode();

	////////////////////////////////////////////////
	//	point 
	////////////////////////////////////////////////

	MFVec2f *getPointField() const;

	void addPoint(float point[]);
	int getNPoints() const;
	void getPoint(int index, float point[]) const;

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

	TextureCoordinateNode *next() const;
	TextureCoordinateNode *nextTraversal() const;

};

}

#endif

