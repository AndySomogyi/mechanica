/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PlaneSensorNode.cpp
*
*	Revisions:
*
*	12/08/02
*		- Changed the super class from SensorNode to DragSensorNode.
*		- Moved the following fields to DragSensorNode.
*			autoOffset, trackPoint
*
******************************************************************/

#include <x3d/PlaneSensorNode.h>

using namespace CyberX3D;

PlaneSensorNode::PlaneSensorNode() 
{
	setHeaderFlag(false);
	setType(PLANESENSOR_NODE);

	// minPosition exposed field
	minPositionField = new SFVec2f(0.0f, 0.0f);
	addExposedField(minPositionFieldString, minPositionField);

	// maxAngle exposed field
	maxPositionField = new SFVec2f(-1.0f, -1.0f);
	addExposedField(maxPositionFieldString, maxPositionField);

	// offset exposed field
	offsetField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addExposedField(offsetFieldString, offsetField);
	
	// translation eventOut field
	translationField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(translationFieldString, translationField);
}

PlaneSensorNode::~PlaneSensorNode() 
{
}

////////////////////////////////////////////////
//	MinPosition
////////////////////////////////////////////////

SFVec2f *PlaneSensorNode::getMinPositionField() const
{
	if (isInstanceNode() == false)
		return minPositionField;
	return (SFVec2f *)getExposedField(minPositionFieldString);
}
	
void PlaneSensorNode::setMinPosition(float value[]) 
{
	getMinPositionField()->setValue(value);
}

void PlaneSensorNode::setMinPosition(float x, float y) 
{
	getMinPositionField()->setValue(x, y);
}

void PlaneSensorNode::getMinPosition(float value[]) const
{
	getMinPositionField()->getValue(value);
}

void PlaneSensorNode::getMinPosition(float *x, float *y) const
{
	SFVec2f *sfvec2f = getMinPositionField();
	*x = sfvec2f->getX();
	*y = sfvec2f->getY();
}

////////////////////////////////////////////////
//	MaxPosition
////////////////////////////////////////////////

SFVec2f *PlaneSensorNode::getMaxPositionField() const
{
	if (isInstanceNode() == false)
		return maxPositionField;
	return (SFVec2f *)getExposedField(maxPositionFieldString);
}
	
void PlaneSensorNode::setMaxPosition(float value[]) 
{
	getMaxPositionField()->setValue(value);
}

void PlaneSensorNode::setMaxPosition(float x, float y) 
{
	getMaxPositionField()->setValue(x, y);
}

void PlaneSensorNode::getMaxPosition(float value[]) const
{
	getMaxPositionField()->getValue(value);
}

void PlaneSensorNode::getMaxPosition(float *x, float *y) const
{
	SFVec2f *sfvec2f = getMaxPositionField();
	*x = sfvec2f->getX();
	*y = sfvec2f->getY();
}

////////////////////////////////////////////////
//	Offset
////////////////////////////////////////////////

SFVec3f *PlaneSensorNode::getOffsetField() const
{
	if (isInstanceNode() == false)
		return offsetField;
	return (SFVec3f *)getExposedField(offsetFieldString);
}
	
void PlaneSensorNode::setOffset(float value[]) 
{
	getOffsetField()->setValue(value);
}

void PlaneSensorNode::getOffset(float value[]) const
{
	getOffsetField()->getValue(value);
}

////////////////////////////////////////////////
//	Translation
////////////////////////////////////////////////

SFVec3f *PlaneSensorNode::getTranslationChangedField() const
{
	if (isInstanceNode() == false)
		return translationField;
	return (SFVec3f *)getEventOut(translationFieldString);
}
	
void PlaneSensorNode::setTranslationChanged(float value[]) 
{
	getTranslationChangedField()->setValue(value);
}

void PlaneSensorNode::setTranslationChanged(float x, float y, float z) 
{
	getTranslationChangedField()->setValue(x, y, z);
}

void PlaneSensorNode::getTranslationChanged(float value[]) const
{
	getTranslationChangedField()->getValue(value);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

PlaneSensorNode *PlaneSensorNode::next() const
{
	return (PlaneSensorNode *)Node::next(getType());
}

PlaneSensorNode *PlaneSensorNode::nextTraversal() const
{
	return (PlaneSensorNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool PlaneSensorNode::isChildNodeType(Node *node) const
{
	return false;
}

void PlaneSensorNode::initialize() 
{
	setIsActive(false);
}

void PlaneSensorNode::uninitialize() 
{
}

void PlaneSensorNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void PlaneSensorNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFBool *autoOffset = getAutoOffsetField();
	const SFBool *enabled = getEnabledField();
	const SFVec2f *maxpos = getMaxPositionField();
	const SFVec2f *minpos = getMinPositionField();
	const SFVec3f *offset = getOffsetField();

	printStream << indentString << "\t" << "autoOffset " << autoOffset  << std::endl;
	printStream << indentString << "\t" << "enabled " << enabled  << std::endl;
	printStream << indentString << "\t" << "maxPosition " << maxpos  << std::endl;
	printStream << indentString << "\t" << "minPosition " << minpos  << std::endl;
	printStream << indentString << "\t" << "offset " << offset << std::endl;
}	
