/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	EnvironmentalSensorNode.cpp
*
*	Revisions:
*
*	12/08/02
*		- The first revision.
*
******************************************************************/

#include <x3d/EnvironmentalSensorNode.h>

using namespace CyberX3D;

EnvironmentalSensorNode::EnvironmentalSensorNode() 
{
	setHeaderFlag(false);
	setType(VISIBILITYSENSOR_NODE);

	// center exposed field
	centerField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addExposedField(centerFieldString, centerField);

	// size exposed field
	sizeField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addExposedField(sizeFieldString, sizeField);

	// enterTime eventOut field
	enterTimeField = new SFTime(0.0f);
	addEventOut(enterTimeFieldString, enterTimeField);

	// exitTime eventOut field
	exitTimeField = new SFTime(0.0f);
	addEventOut(exitTimeFieldString, exitTimeField);
}

EnvironmentalSensorNode::~EnvironmentalSensorNode() 
{
}

////////////////////////////////////////////////
//	Center
////////////////////////////////////////////////

SFVec3f *EnvironmentalSensorNode::getCenterField() const
{
	if (isInstanceNode() == false)
		return centerField;
	return (SFVec3f *)getExposedField(centerFieldString);
}
	
void EnvironmentalSensorNode::setCenter(float value[]) 
{
	getCenterField()->setValue(value);
}

void EnvironmentalSensorNode::setCenter(float x, float y, float z) 
{
	getCenterField()->setValue(x, y, z);
}

void EnvironmentalSensorNode::getCenter(float value[]) const 
{
	getCenterField()->getValue();
}

////////////////////////////////////////////////
//	Size
////////////////////////////////////////////////

SFVec3f *EnvironmentalSensorNode::getSizeField() const
{
	if (isInstanceNode() == false)
		return sizeField;
	return (SFVec3f *)getExposedField(sizeFieldString);
}
	
void EnvironmentalSensorNode::setSize(float value[]) 
{
	getSizeField()->setValue(value);
}

void EnvironmentalSensorNode::setSize(float x, float y, float z) 
{
	getSizeField()->setValue(x, y, z);
}

void EnvironmentalSensorNode::getSize(float value[]) const 
{
	getSizeField()->getValue();
}

////////////////////////////////////////////////
//	EnterTime
////////////////////////////////////////////////

SFTime *EnvironmentalSensorNode::getEnterTimeField() const
{
	if (isInstanceNode() == false)
		return enterTimeField;
	return (SFTime *)getEventOut(enterTimeFieldString);
}
	
void EnvironmentalSensorNode::setEnterTime(double value) 
{
	getEnterTimeField()->setValue(value);
}

double EnvironmentalSensorNode::getEnterTime() const 
{
	return getEnterTimeField()->getValue();
}

////////////////////////////////////////////////
//	ExitTime
////////////////////////////////////////////////

SFTime *EnvironmentalSensorNode::getExitTimeField() const
{
	if (isInstanceNode() == false)
		return exitTimeField;
	return (SFTime *)getEventOut(exitTimeFieldString);
}
	
void EnvironmentalSensorNode::setExitTime(double value) 
{
	getExitTimeField()->setValue(value);
}

double EnvironmentalSensorNode::getExitTime() const 
{
	return getExitTimeField()->getValue();
}

