/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TransformNode.h
*
******************************************************************/

#ifndef _CX3D_TRANSFORMNODE_H_
#define _CX3D_TRANSFORMNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>
#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {

class TransformNode : public BoundedGroupingNode {

	SFVec3f *translationField;
	SFVec3f *scaleField;
	SFVec3f *centerField;
	SFRotation *rotationField;
	SFRotation *scaleOrientationField;

public:

	TransformNode();
	virtual ~TransformNode();

	////////////////////////////////////////////////
	//	Translation
	////////////////////////////////////////////////

	SFVec3f *getTranslationField() const;

	void setTranslation(float value[]);
	void setTranslation(float x, float y, float z);
	void getTranslation(float value[]) const;

	////////////////////////////////////////////////
	//	Scale
	////////////////////////////////////////////////

	SFVec3f *getScaleField() const;

	void setScale(float value[]);
	void setScale(float x, float y, float z);
	void getScale(float value[]) const;

	////////////////////////////////////////////////
	//	Center
	////////////////////////////////////////////////

	SFVec3f *getCenterField() const;

	void setCenter(float value[]);
	void setCenter(float x, float y, float z);
	void getCenter(float value[]) const;

	////////////////////////////////////////////////
	//	Rotation
	////////////////////////////////////////////////

	SFRotation *getRotationField() const;

	void setRotation(float value[]);
	void setRotation(float x, float y, float z, float w);
	void getRotation(float value[]) const;

	////////////////////////////////////////////////
	//	ScaleOrientation
	////////////////////////////////////////////////

	SFRotation *getScaleOrientationField() const;

	void setScaleOrientation(float value[]);
	void setScaleOrientation(float x, float y, float z, float w);
	void getScaleOrientation(float value[]) const;

	////////////////////////////////////////////////
	//	Matrix
	////////////////////////////////////////////////

	void getSFMatrix(SFMatrix *mOut) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TransformNode *next() const;
	TransformNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Infomation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

