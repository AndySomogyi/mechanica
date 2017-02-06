/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Polyline2DNode.h
*
******************************************************************/

#ifndef _CX3D_POLYLINE2D_H_
#define _CX3D_POLYLINE2D_H_

#include <x3d/Geometry2DNode.h>

namespace CyberX3D {

class Polyline2DNode : public Geometry2DNode {

	MFVec2f *lineSegmentsField;

public:

	Polyline2DNode();
	virtual ~Polyline2DNode();

	////////////////////////////////////////////////
	//	LineSegments
	////////////////////////////////////////////////

	MFVec2f *getLineSegmentsField() const;
	
	int getNLineSegments() const;
	void addLineSegment(float point[]);
	void addLineSegment(float x, float y);
	void getLineSegment(int index, float point[]) const;
	void setLineSegment(int index, float point[]);
	void setLineSegment(int index, float x, float y);
	void removeLineSegment(int index);
	void removeAllLineSegments();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	Polyline2DNode *next() const;
	Polyline2DNode *nextTraversal() const;

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
