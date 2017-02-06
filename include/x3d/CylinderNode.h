/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CylinderNode.h
*
******************************************************************/

#ifndef _CX3D_CYLINDER_H_
#define _CX3D_CYLINDER_H_

#include <x3d/Geometry3DNode.h>

namespace CyberX3D {

const int DEFAULT_CYLINDERNODE_SLICES = 12;

class CylinderNode : public Geometry3DNode {

	SFFloat *radiusField;
	SFFloat *heightField;
	SFBool *topField;
	SFBool *sideField;
	SFBool *bottomField;

public:

	CylinderNode();
	virtual ~CylinderNode();

	////////////////////////////////////////////////
	//	radius
	////////////////////////////////////////////////

	SFFloat *getRadiusField() const;

	void setRadius(float value);
	float getRadius() const;

	////////////////////////////////////////////////
	//	height
	////////////////////////////////////////////////

	SFFloat *getHeightField() const;

	void setHeight(float value);
	float getHeight() const;

	////////////////////////////////////////////////
	//	top
	////////////////////////////////////////////////

	SFBool *getTopField() const;

	void setTop(bool value);
	void setTop(int value);
	bool getTop() const;

	////////////////////////////////////////////////
	//	side
	////////////////////////////////////////////////

	SFBool *getSideField() const;

	void setSide(bool value);
	void setSide(int value);
	bool getSide() const;

	////////////////////////////////////////////////
	//	bottom
	////////////////////////////////////////////////

	SFBool *getBottomField() const;

	void setBottom(bool  value);
	void setBottom(int value);
	bool  getBottom() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	CylinderNode *next() const;
	CylinderNode *nextTraversal() const;

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

	int getNPolygons() const;

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

