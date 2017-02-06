/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SensorNode.h
*
******************************************************************/

#ifndef _CX3D_SENSORNODE_H_
#define _CX3D_SENSORNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class SensorNode : public Node {
	
	SFBool *enabledField;
	SFBool *isActiveField;

public:

	SensorNode();
	virtual ~SensorNode();

	////////////////////////////////////////////////
	//	Enabled
	////////////////////////////////////////////////

	SFBool *getEnabledField() const;
	
	void setEnabled(bool  value);
	void setEnabled(int value);
	bool  getEnabled() const;
	bool  isEnabled() const;

	////////////////////////////////////////////////
	//	isActive
	////////////////////////////////////////////////
	
	SFBool *getIsActiveField() const;

	void setIsActive(bool  value);
	void setIsActive(int value);
	bool  getIsActive() const;
	bool isActive() const;

};

}

#endif

