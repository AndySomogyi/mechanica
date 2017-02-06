/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Circle2DNode.h
*
******************************************************************/

#ifndef _CX3D_CIRCLE2D_H_
#define _CX3D_CIRCLE2D_H_

#include <x3d/Geometry2DNode.h>

namespace CyberX3D {

class Circle2DNode : public Geometry2DNode {

	SFFloat *radiusField;

public:

	Circle2DNode();
	virtual ~Circle2DNode();

	////////////////////////////////////////////////
	//	Radius
	////////////////////////////////////////////////

	SFFloat *getRadiusField() const;
	
	void setRadius(float value);
	float getRadius() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	Circle2DNode *next() const;
	Circle2DNode *nextTraversal() const;

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

