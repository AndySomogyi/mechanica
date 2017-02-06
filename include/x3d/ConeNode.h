/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ConeNode.h
*
******************************************************************/

#ifndef _CX3D_CONE_H_
#define _CX3D_CONE_H_

#include <x3d/Geometry3DNode.h>

namespace CyberX3D {

const int DEFAULT_CONENODE_SLICES = 15;

class ConeNode : public Geometry3DNode {

	SFFloat *bottomRadiusField;
	SFFloat *heightField;
	SFBool *sideField;
	SFBool *bottomField;

public:

	ConeNode();
	virtual ~ConeNode();

	////////////////////////////////////////////////
	//	bottomRadius
	////////////////////////////////////////////////

	SFFloat *getBottomRadiusField() const;

	void setBottomRadius(float value);
	float getBottomRadius() const;

	////////////////////////////////////////////////
	//	height
	////////////////////////////////////////////////

	SFFloat *getHeightField() const;

	void setHeight(float value);
	float getHeight() const;

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

	void setBottom(bool value);
	void setBottom(int value);
	bool getBottom() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ConeNode *next() const;
	ConeNode *nextTraversal() const;

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

