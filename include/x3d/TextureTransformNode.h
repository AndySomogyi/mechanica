/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TextureTransformNode.h
*
******************************************************************/

#ifndef _CX3D_TEXTURETRANSFORMNODE_H_
#define _CX3D_TEXTURETRANSFORMNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/AppearanceChildNode.h>

namespace CyberX3D {

class TextureTransformNode : public AppearanceChildNode {

	SFVec2f *translationField;
	SFVec2f *scaleField;
	SFVec2f *centerField;
	SFFloat *rotationField;

public:

	TextureTransformNode();
	virtual ~TextureTransformNode();

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
	//	Center
	////////////////////////////////////////////////

	SFVec2f *getCenterField() const;

	void setCenter(float value[]);
	void setCenter(float x, float y);
	void getCenter(float value[]) const;

	////////////////////////////////////////////////
	//	Rotation
	////////////////////////////////////////////////

	SFFloat *getRotationField() const;

	void setRotation(float value);
	float getRotation() const;

	////////////////////////////////////////////////
	//	Texture Matrix
	////////////////////////////////////////////////

	void getSFMatrix(SFMatrix *matrix) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TextureTransformNode *next() const;
	TextureTransformNode *nextTraversal() const;

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

