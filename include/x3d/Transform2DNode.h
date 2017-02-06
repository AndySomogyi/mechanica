/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Transform2DNode.h
*
******************************************************************/

#ifndef _CX3D_TRANSFORM2DNODE_H_
#define _CX3D_TRANSFORM2DNODE_H_

#include <x3d/BoundedGrouping2DNode.h>

namespace CyberX3D {

class Transform2DNode : public BoundedGrouping2DNode {

	SFVec2f *translationField;
	SFVec2f *scaleField;
	SFRotation *rotationField;
	SFRotation *scaleOrientationField;

public:

	Transform2DNode();
	virtual ~Transform2DNode();

	////////////////////////////////////////////////
	//	Translation
	////////////////////////////////////////////////

	SFVec2f *getTranslationField() const;

	void setTranslation(float value[]);
	void setTranslation(float x, float y);
	void getTranslation(float value[]) const;

	////////////////////////////////////////////////
	//	Scale
	////////////////////////////////////////////////

	SFVec2f *getScaleField() const;

	void setScale(float value[]);
	void setScale(float x, float y);
	void getScale(float value[]) const;

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

	Transform2DNode *next() const;
	Transform2DNode *nextTraversal() const;

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
