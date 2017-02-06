/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ExtrusionNode.h
*
******************************************************************/

#ifndef _CX3D_EXTRUSIONNODE_H_
#define _CX3D_EXTRUSIONNODE_H_

#include <x3d/Geometry3DNode.h>

namespace CyberX3D {

class ExtrusionNode : public Geometry3DNode 
{
	SFBool *beginCapField;
	SFBool *endCapField;
	SFBool *convexField;
	SFFloat *creaseAngleField;
	SFBool *ccwField;
	SFBool *solidField;
	MFRotation *orientationField;
	MFVec2f *scaleField;
	MFVec2f *crossSectionField;
	MFVec3f *spineField;

public:

	ExtrusionNode();
	virtual ~ExtrusionNode();

	////////////////////////////////////////////////
	//	BeginCap
	////////////////////////////////////////////////
	
	SFBool *getBeginCapField() const;

	void setBeginCap(bool value);
	void setBeginCap(int value);
	bool getBeginCap() const;

	////////////////////////////////////////////////
	//	EndCap
	////////////////////////////////////////////////

	SFBool *getEndCapField() const;
	
	void setEndCap(bool value);
	void setEndCap(int value);
	bool getEndCap() const;

	////////////////////////////////////////////////
	//	Convex
	////////////////////////////////////////////////

	SFBool *getConvexField() const;
	
	void setConvex(bool value);
	void setConvex(int value);
	bool getConvex() const;

	////////////////////////////////////////////////
	//	CCW
	////////////////////////////////////////////////

	SFBool *getCCWField() const;
	
	void setCCW(bool value);
	void setCCW(int value);
	bool getCCW() const;

	////////////////////////////////////////////////
	//	Solid
	////////////////////////////////////////////////

	SFBool *getSolidField() const;
	
	void setSolid(bool value);
	void setSolid(int value);
	bool getSolid() const;

	////////////////////////////////////////////////
	//	CreaseAngle
	////////////////////////////////////////////////

	SFFloat *getCreaseAngleField() const;
	
	void setCreaseAngle(float value);
	float getCreaseAngle() const;

	////////////////////////////////////////////////
	// orientation
	////////////////////////////////////////////////

	MFRotation *getOrientationField() const;

	void addOrientation(float value[]);
	void addOrientation(float x, float y, float z, float angle);
	int getNOrientations() const;
	void getOrientation(int index, float value[]) const;

	////////////////////////////////////////////////
	// scale
	////////////////////////////////////////////////

	MFVec2f *getScaleField() const;

	void addScale(float value[]);
	void addScale(float x, float z);
	int getNScales() const;
	void getScale(int index, float value[]) const;

	////////////////////////////////////////////////
	// crossSection
	////////////////////////////////////////////////

	MFVec2f *getCrossSectionField() const;

	void addCrossSection(float value[]);
	void addCrossSection(float x, float z);
	int getNCrossSections() const;
	void getCrossSection(int index, float value[]) const;

	////////////////////////////////////////////////
	// spine
	////////////////////////////////////////////////

	MFVec3f *getSpineField() const;

	void addSpine(float value[]);
	void addSpine(float x, float y, float z);
	int getNSpines() const;
	void getSpine(int index, float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	ExtrusionNode *next() const;
	ExtrusionNode *nextTraversal() const;

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
	//	recomputeDisplayList
	////////////////////////////////////////////////

	void recomputeDisplayList();

	////////////////////////////////////////////////
	//	Polygons
	////////////////////////////////////////////////

	int getNPolygons() const;

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

