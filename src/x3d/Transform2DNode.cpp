/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	Transform2DNode.cpp
*
******************************************************************/

#include <x3d/X3DFields.h>
#include <x3d/Transform2DNode.h>

using namespace CyberX3D;

Transform2DNode::Transform2DNode() 
{
	setHeaderFlag(false);
	setType(TRANSFORM2D_NODE);

	// translation exposed field
	translationField = new SFVec2f(0.0f, 0.0f);
	translationField->setName(translationFieldString);
	addExposedField(translationField);

	// scale exposed field
	scaleField = new SFVec2f(1.0f, 1.0f);
	scaleField->setName(scaleFieldString);
	addExposedField(scaleField);

	// rotation exposed field
	rotationField = new SFRotation(0.0f, 0.0f, 1.0f, 0.0f);
	rotationField->setName(rotationFieldString);
	addExposedField(rotationField);

	// scaleOrientation exposed field
	scaleOrientationField = new SFRotation(0.0f, 0.0f, 1.0f, 0.0f);
	scaleOrientationField->setName(scaleOrientationFieldString);
	addExposedField(scaleOrientationField);
}

Transform2DNode::~Transform2DNode() 
{
}

////////////////////////////////////////////////
//	Translation
////////////////////////////////////////////////

SFVec2f *Transform2DNode::getTranslationField() const
{
	if (isInstanceNode() == false)
		return translationField;
	return (SFVec2f *)getExposedField(translationFieldString);
}

void Transform2DNode::setTranslation(float value[]) 
{
	getTranslationField()->setValue(value);
}

void Transform2DNode::setTranslation(float x, float y) 
{
	getTranslationField()->setValue(x, y);
}

void Transform2DNode::getTranslation(float value[]) const
{
	getTranslationField()->getValue(value);
}

////////////////////////////////////////////////
//	Scale
////////////////////////////////////////////////

SFVec2f *Transform2DNode::getScaleField() const
{
	if (isInstanceNode() == false)
		return scaleField;
	return (SFVec2f *)getExposedField(scaleFieldString);
}

void Transform2DNode::setScale(float value[]) 
{
	getScaleField()->setValue(value);
}

void Transform2DNode::setScale(float x, float y) 
{
	getScaleField()->setValue(x, y);
}

void Transform2DNode::getScale(float value[]) const
{
	getScaleField()->getValue(value);
}

////////////////////////////////////////////////
//	Rotation
////////////////////////////////////////////////

SFRotation *Transform2DNode::getRotationField() const
{
	if (isInstanceNode() == false)
		return rotationField;
	return (SFRotation *)getExposedField(rotationFieldString);
}

void Transform2DNode::setRotation(float value[]) 
{
	getRotationField()->setValue(value);
}

void Transform2DNode::setRotation(float x, float y, float z, float w) 
{
	getRotationField()->setValue(x, y, z, w);
}

void Transform2DNode::getRotation(float value[]) const
{
	getRotationField()->getValue(value);
}

////////////////////////////////////////////////
//	ScaleOrientation
////////////////////////////////////////////////

SFRotation *Transform2DNode::getScaleOrientationField() const
{
	if (isInstanceNode() == false)
		return scaleOrientationField;
	return (SFRotation *)getExposedField(scaleOrientationFieldString);
}

void Transform2DNode::setScaleOrientation(float value[]) 
{
	getScaleOrientationField()->setValue(value);
}

void Transform2DNode::setScaleOrientation(float x, float y, float z, float w) 
{
	getScaleOrientationField()->setValue(x, y, z, w);
}

void Transform2DNode::getScaleOrientation(float value[]) const
{
	getScaleOrientationField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

Transform2DNode *Transform2DNode::next() const
{
	return (Transform2DNode *)Node::next(getType());
}

Transform2DNode *Transform2DNode::nextTraversal() const
{
	return (Transform2DNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////

bool Transform2DNode::isChildNodeType(Node *node) const
{
	if (node->isCommonNode() || node->isBindableNode() ||node->isInterpolatorNode() || node->isSensorNode() || node->isGroupingNode() || node->isSpecialGroupNode())
		return true;
	else
		return false;
}

void Transform2DNode::initialize() 
{
	recomputeBoundingBox();
}

void Transform2DNode::uninitialize() 
{
}

void Transform2DNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void Transform2DNode::outputContext(std::ostream &printStream, const char *indentString) const
{
}

////////////////////////////////////////////////
//	Matrix
////////////////////////////////////////////////

void Transform2DNode::getSFMatrix(SFMatrix *mOut) const
{
	float	rotation[4];
	float	scale[3];
	float	scaleOri[4];
	float	trans[3];
	SFMatrix	mSRI;
	SFMatrix	mSR;
	SFMatrix	mT;
	SFMatrix	mR;
	SFMatrix	mS;

	getTranslation(trans); 
	trans[2] = 0.0f;
	mT.setTranslation(trans);

	getRotation(rotation);
	mR.setRotation(rotation);

	getScaleOrientation(scaleOri); 
	mSR.setRotation(scaleOri);

	getScale(scale);
	trans[2] = 1.0f;
	mS.setScaling(scale);

	getScaleOrientation(scaleOri); 
	scaleOri[3] = -scaleOri[3]; 
	mSRI.setRotation(scaleOri);

	mOut->init();
	mOut->add(&mT);
	mOut->add(&mR);
	mOut->add(&mSR);
	mOut->add(&mS);
	mOut->add(&mSRI);
}
