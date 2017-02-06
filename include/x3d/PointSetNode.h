/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PointSetNode.h
*
******************************************************************/

#ifndef _CX3D_POINTSET_H_
#define _CX3D_POINTSET_H_

#include <x3d/Geometry3DNode.h>
#include <x3d/ColorNode.h>
#include <x3d/CoordinateNode.h>

namespace CyberX3D {

class PointSetNode : public Geometry3DNode {

public:

	PointSetNode();
	virtual ~PointSetNode();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	PointSetNode *next() const;
	PointSetNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	void recomputeBoundingBox();

	////////////////////////////////////////////////
	//	Polygons
	////////////////////////////////////////////////

	int getNPolygons() const {
		return 0;
	}

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

