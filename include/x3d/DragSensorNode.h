/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	DragSensorNode.h
*
******************************************************************/

#ifndef _CX3D_DRAGSENSORNODE_H_
#define _CX3D_DRAGSENSORNODE_H_

#include <x3d/PointingDeviceSensorNode.h>

namespace CyberX3D {

class DragSensorNode : public PointingDeviceSensorNode
{
	SFBool *autoOffsetField;
	SFVec3f *trackPointField;

public:

	DragSensorNode();
	virtual ~DragSensorNode();

	////////////////////////////////////////////////
	//	AutoOffset
	////////////////////////////////////////////////

	SFBool *getAutoOffsetField() const;
	
	void setAutoOffset(bool  value);
	void setAutoOffset(int value);
	bool  getAutoOffset() const;
	bool  isAutoOffset() const;

	////////////////////////////////////////////////
	//	TrackPoint
	////////////////////////////////////////////////

	SFVec3f *getTrackPointChangedField() const;
	
	void setTrackPointChanged(float value[]);
	void setTrackPointChanged(float x, float y, float z);
	void getTrackPointChanged(float value[]) const;

};

}

#endif

