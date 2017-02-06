/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DragSensorNode.cpp
*
*	Revisions:
*
*	12/07/02
*		- The first revision.
*
******************************************************************/

#include <x3d/DragSensorNode.h>

using namespace CyberX3D;

DragSensorNode::DragSensorNode() 
{
	// autoOffset exposed field
	autoOffsetField = new SFBool(true);
	addExposedField(autoOffsetFieldString, autoOffsetField);

	// trackPoint eventOut field
	trackPointField = new SFVec3f(0.0f, 0.0f, 0.0f);
	addEventOut(trackPointFieldString, trackPointField);
}

DragSensorNode::~DragSensorNode() 
{
}

////////////////////////////////////////////////
//	AutoOffset
////////////////////////////////////////////////

SFBool *DragSensorNode::getAutoOffsetField() const
{
	if (isInstanceNode() == false)
		return autoOffsetField;
	return (SFBool *)getExposedField(autoOffsetFieldString);
}
	
void DragSensorNode::setAutoOffset(bool  value) 
{
	getAutoOffsetField()->setValue(value);
}

void DragSensorNode::setAutoOffset(int value) 
{
	setAutoOffset(value ? true : false);
}

bool  DragSensorNode::getAutoOffset() const
{
	return getAutoOffsetField()->getValue();
}

bool  DragSensorNode::isAutoOffset() const 
{
	return getAutoOffset();
}

////////////////////////////////////////////////
//	TrackPoint
////////////////////////////////////////////////

SFVec3f *DragSensorNode::getTrackPointChangedField() const
{
	if (isInstanceNode() == false)
		return trackPointField;
	return (SFVec3f *)getEventOut(trackPointFieldString);
}
	
void DragSensorNode::setTrackPointChanged(float value[]) 
{
	getTrackPointChangedField()->setValue(value);
}

void DragSensorNode::setTrackPointChanged(float x, float y, float z) 
{
	getTrackPointChangedField()->setValue(x, y, z);
}

void DragSensorNode::getTrackPointChanged(float value[]) const 
{
	getTrackPointChangedField()->getValue(value);
}
