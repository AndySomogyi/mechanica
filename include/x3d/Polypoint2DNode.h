/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Polypoint2DNode.h
*
******************************************************************/

#ifndef _CX3D_POLYPOINT2D_H_
#define _CX3D_POLYPOINT2D_H_

#include <x3d/Geometry2DNode.h>

namespace CyberX3D {

class Polypoint2DNode : public Geometry2DNode {

	MFVec2f *pointsField;

public:

	Polypoint2DNode();
	virtual ~Polypoint2DNode();

	////////////////////////////////////////////////
	//	Points
	////////////////////////////////////////////////

	MFVec2f *getPointsField() const;
	
	int getNPoints() const;
	void addPoint(float point[]);
	void addPoint(float x, float y);
	void getPoint(int index, float point[]) const;
	void setPoint(int index, float point[]);
	void setPoint(int index, float x, float y);
	void removePoint(int index);
	void removeAllPoints();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	Polypoint2DNode *next() const;
	Polypoint2DNode *nextTraversal() const;

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
