/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	KeyDeviceSensorNode.h
*
******************************************************************/

#ifndef _CX3D_KEYDEVICESENSORNODE_H_
#define _CX3D_KEYDEVICESENSORNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/SensorNode.h>

namespace CyberX3D {

class KeyDeviceSensorNode : public SensorNode {
	
	SFInt32 *actionKeyPressField;
	SFInt32 *actionKeyReleaseField;
	SFBool *altKeyField;
	SFBool *controlKeyField;
	SFBool *shiftKeyField;
	SFInt32 *keyPressField;
	SFInt32 *keyReleaseField;

public:

	KeyDeviceSensorNode();
	virtual ~KeyDeviceSensorNode();

	////////////////////////////////////////////////
	//	AltKey
	////////////////////////////////////////////////

	SFBool *getAltKeyField() const;
	
	void setAltKey(bool value);
	void setAltKey(int value);
	bool getAltKey() const;

	////////////////////////////////////////////////
	//	ControlKey
	////////////////////////////////////////////////

	SFBool *getControlKeyField() const;
	
	void setControlKey(bool value);
	void setControlKey(int value);
	bool getControlKey() const;

	////////////////////////////////////////////////
	//	ShiftKey
	////////////////////////////////////////////////

	SFBool *getShiftKeyField() const;
	
	void setShiftKey(bool value);
	void setShiftKey(int value);
	bool getShiftKey() const;

	////////////////////////////////////////////////
	//	ActionKeyPress
	////////////////////////////////////////////////

	SFInt32 *getActionKeyPressField() const;
	
	void setActionKeyPress(int value);
	int getActionKeyPress() const;

	////////////////////////////////////////////////
	//	ActionKeyRelease
	////////////////////////////////////////////////

	SFInt32 *getActionKeyReleaseField() const;
	
	void setActionKeyRelease(int value);
	int getActionKeyRelease() const;

	////////////////////////////////////////////////
	//	KeyPress
	////////////////////////////////////////////////

	SFInt32 *getKeyPressField() const;
	
	void setKeyPress(int value);
	int getKeyPress() const;

	////////////////////////////////////////////////
	//	KeyRelease
	////////////////////////////////////////////////

	SFInt32 *getKeyReleaseField() const;
	
	void setKeyRelease(int value);
	int getKeyRelease() const;
};

}

#endif

