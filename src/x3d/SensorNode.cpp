/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SensorNode.cpp
*
******************************************************************/

#include <x3d/SensorNode.h>

using namespace CyberX3D;

SensorNode::SensorNode() 
{
	// enabled exposed field
	enabledField = new SFBool(true);
	addExposedField(enabledFieldString, enabledField);

	// isActive eventOut field
	isActiveField = new SFBool(false);
	addEventOut(isActiveFieldString, isActiveField);
}

SensorNode::~SensorNode() 
{
}

////////////////////////////////////////////////
//	Enabled
////////////////////////////////////////////////

SFBool *SensorNode::getEnabledField() const
{
	if (isInstanceNode() == false)
		return enabledField;
	return (SFBool *)getExposedField(enabledFieldString);
}
	
void SensorNode::setEnabled(bool  value) 
{
	getEnabledField()->setValue(value);
}

void SensorNode::setEnabled(int value) 
{
	setEnabled(value ? true : false);
}

bool  SensorNode::getEnabled() const
{
	return getEnabledField()->getValue();
}

bool  SensorNode::isEnabled() const
{
	return getEnabled();
}

////////////////////////////////////////////////
//	isActive
////////////////////////////////////////////////

SFBool *SensorNode::getIsActiveField() const
{
	if (isInstanceNode() == false)
		return isActiveField;
	return (SFBool *)getEventOut(isActiveFieldString);
}
	
void SensorNode::setIsActive(bool  value) 
{
	getIsActiveField()->setValue(value);
}

void SensorNode::setIsActive(int value) 
{
	setIsActive(value ? true : false);
}

bool  SensorNode::getIsActive() const
{
	return getIsActiveField()->getValue();
}

bool SensorNode::isActive() const
{
	return getIsActive();
}

