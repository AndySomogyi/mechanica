/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TouchSensorNode.cpp
*
******************************************************************/

#include <x3d/TouchSensorNode.h>

using namespace CyberX3D;

TouchSensorNode::TouchSensorNode() 
{
	setHeaderFlag(false);
	setType(TOUCHSENSOR_NODE);

	// hitNormal eventOut field
	hitNormalField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(hitNormalFieldString, hitNormalField);

	// hitTexCoord eventOut field
	hitTexCoordField = new SFVec2f(0.0f, 0.0f);
	addEventOut(hitTexCoordFieldString, hitTexCoordField);

	// hitPoint eventOut field
	hitPointField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(hitPointFieldString, hitPointField);

	// isOver eventOut field
	isOverField = new SFBool(false);
	addEventOut(isOverFieldString, isOverField);

	// exitTime eventOut field
	touchTimeField = new SFTime(0.0f);
	addEventOut(touchTimeFieldString, touchTimeField);
}

TouchSensorNode::~TouchSensorNode() 
{
}

////////////////////////////////////////////////
//	isOver
////////////////////////////////////////////////

SFBool *TouchSensorNode::getIsOverField() const
{
	if (isInstanceNode() == false)
		return isOverField;
	return (SFBool *)getEventOut(isOverFieldString);
}
	
void TouchSensorNode::setIsOver(bool  value) 
{
	getIsOverField()->setValue(value);
}

void TouchSensorNode::setIsOver(int value) 
{
	setIsOver(value ? true : false);
}

bool  TouchSensorNode::getIsOver() const
{
	return getIsOverField()->getValue();
}

bool  TouchSensorNode::isOver() const
{
	return getIsOver();
}

////////////////////////////////////////////////
//	hitNormal
////////////////////////////////////////////////

SFVec3f *TouchSensorNode::getHitNormalChangedField() const
{
	if (isInstanceNode() == false)
		return hitNormalField;
	return (SFVec3f *)getEventOut(hitNormalFieldString);
}
	
void TouchSensorNode::setHitNormalChanged(float value[]) 
{
	getHitNormalChangedField()->setValue(value);
}

void TouchSensorNode::setHitNormalChanged(float x, float y, float z) 
{
	getHitNormalChangedField()->setValue(x, y, z);
}

void TouchSensorNode::getHitNormalChanged(float value[]) const
{
	getHitNormalChangedField()->getValue(value);
}

////////////////////////////////////////////////
//	hitPoint
////////////////////////////////////////////////

SFVec3f *TouchSensorNode::getHitPointChangedField() const
{
	if (isInstanceNode() == false)
		return hitPointField;
	return (SFVec3f *)getEventOut(hitPointFieldString);
}
	
void TouchSensorNode::setHitPointChanged(float value[]) 
{
	getHitPointChangedField()->setValue(value);
}

void TouchSensorNode::setHitPointChanged(float x, float y, float z) 
{
	getHitPointChangedField()->setValue(x, y, z);
}

void TouchSensorNode::getHitPointChanged(float value[]) const
{
	getHitPointChangedField()->getValue(value);
}

////////////////////////////////////////////////
//	hitTexCoord
////////////////////////////////////////////////

SFVec2f *TouchSensorNode::getHitTexCoordField() const
{
	if (isInstanceNode() == false)
		return hitTexCoordField;
	return (SFVec2f *)getEventOut(hitTexCoordFieldString);
}
	
void TouchSensorNode::setHitTexCoord(float value[]) 
{
	getHitTexCoordField()->setValue(value);
}

void TouchSensorNode::setHitTexCoord(float x, float y) 
{
	getHitTexCoordField()->setValue(x, y);
}

void TouchSensorNode::getHitTexCoord(float value[]) const
{
	getHitTexCoordField()->getValue(value);
}

////////////////////////////////////////////////
//	ExitTime
////////////////////////////////////////////////

SFTime *TouchSensorNode::getTouchTimeField() const
{
	if (isInstanceNode() == false)
		return touchTimeField;
	return (SFTime *)getEventOut(touchTimeFieldString);
}
	
void TouchSensorNode::setTouchTime(double value) 
{
	getTouchTimeField()->setValue(value);
}

double TouchSensorNode::getTouchTime() const
{
	return getTouchTimeField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TouchSensorNode *TouchSensorNode::next() const
{
	return (TouchSensorNode *)Node::next(getType());
}

TouchSensorNode *TouchSensorNode::nextTraversal() const
{
	return (TouchSensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool TouchSensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void TouchSensorNode::initialize() 
{
}

void TouchSensorNode::uninitialize() 
{
}

void TouchSensorNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void TouchSensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	SFBool *enabled = getEnabledField();
	printStream << indentString << "\t" << "enabled " << enabled << std::endl;
}

