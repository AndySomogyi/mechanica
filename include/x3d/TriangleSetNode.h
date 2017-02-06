/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TriangleSetNode.h
*
******************************************************************/

#ifndef _CX3D_TRIANGLESETNODE_H_
#define _CX3D_TRIANGLESETNODE_H_

#include <x3d/ComposedGeometryNode.h>
#include <x3d/NormalNode.h>
#include <x3d/ColorNode.h>
#include <x3d/CoordinateNode.h>
#include <x3d/TextureCoordinateNode.h>

namespace CyberX3D {

class TriangleSetNode : public ComposedGeometryNode 
{
	SFBool *convexField;
	SFFloat *creaseAngleField;
	
public:

	TriangleSetNode();
	virtual ~TriangleSetNode();
	
	////////////////////////////////////////////////
	//	Convex
	////////////////////////////////////////////////

	SFBool *getConvexField() const;
	
	void setConvex(bool value);
	void setConvex(int value);
	bool getConvex() const;

	////////////////////////////////////////////////
	//	CreaseAngle
	////////////////////////////////////////////////

	SFFloat *getCreaseAngleField() const;
	
	void setCreaseAngle(float value);
	float getCreaseAngle() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TriangleSetNode *next() const;
	TriangleSetNode *nextTraversal() const;

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
	//	Polygon
	////////////////////////////////////////////////

	int getNPolygons() const;

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

};

}

#endif

