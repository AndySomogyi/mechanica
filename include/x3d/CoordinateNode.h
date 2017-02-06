/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CoordinateNode.h
*
******************************************************************/

#ifndef _CX3D_COOORDINATENODE_H_
#define _CX3D_COOORDINATENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/GeometricPropertyNode.h>

namespace CyberX3D {

class CoordinateNode : public GeometricPropertyNode {

	MFVec3f *pointField;

public:

	CoordinateNode();
	virtual ~CoordinateNode();

	////////////////////////////////////////////////
	//	point 
	////////////////////////////////////////////////

	MFVec3f *getPointField() const;

	void addPoint(float point[]);
	void addPoint(float x, float y, float z);
	int getNPoints() const;
	void getPoint(int index, float point[]) const;
	void setPoint(int index, float point[]);
	void setPoint(int index, float x, float y, float z);
	void removePoint(int index);
	void removeLastPoint();
	void removeFirstPoint();
	void removeAllPoints();

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

	CoordinateNode *next() const;
	CoordinateNode *nextTraversal() const;
};

}

#endif

