/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Arc2DNode.h
*
******************************************************************/

#ifndef _CX3D_ARC2DNODE_H_
#define _CX3D_ARC2DNODE_H_

#include <x3d/Geometry2DNode.h>

namespace CyberX3D {

class Arc2DNode : public Geometry2DNode {

	SFFloat *radiusField;
	SFFloat *startAngleField;
	SFFloat *endAngleField;

public:

	Arc2DNode();
	virtual ~Arc2DNode();

	////////////////////////////////////////////////
	//	Radius
	////////////////////////////////////////////////

	SFFloat *getRadiusField() const;
	
	void setRadius(float value);
	float getRadius() const;

	////////////////////////////////////////////////
	//	startAngle
	////////////////////////////////////////////////

	SFFloat *getStartAngleField() const;
	
	void setStartAngle(float value);
	float getStartAngle() const;

	////////////////////////////////////////////////
	//	endAngle
	////////////////////////////////////////////////

	SFFloat *getEndAngleField() const;
	
	void setEndAngle(float value);
	float getEndAngle() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	Arc2DNode *next() const;
	Arc2DNode *nextTraversal() const;

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
