/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Disk2DNode.h
*
******************************************************************/

#ifndef _CX3D_DISK2D_H_
#define _CX3D_DISK2D_H_

#include <x3d/Geometry2DNode.h>

namespace CyberX3D {

class Disk2DNode : public Geometry2DNode {

	SFFloat *outerRadiusField;
	SFFloat *innerRadiusField;

public:

	Disk2DNode();
	virtual ~Disk2DNode();

	////////////////////////////////////////////////
	//	Radius
	////////////////////////////////////////////////

	SFFloat *getOuterRadiusField() const;
	
	void setOuterRadius(float value);
	float getOuterRadius() const;

	////////////////////////////////////////////////
	//	Radius
	////////////////////////////////////////////////

	SFFloat *getInnerRadiusField() const;
	
	void setInnerRadius(float value);
	float getInnerRadius() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	Disk2DNode *next() const;
	Disk2DNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	recomputeDisplayList
	////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL
	void recomputeDisplayList();
#endif

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
