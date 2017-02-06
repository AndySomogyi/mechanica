/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CylinderSensorNode.h
*
******************************************************************/

#ifndef _CX3D_CYLINDERSENSOR_H_
#define _CX3D_CYLINDERSENSOR_H_

#include <x3d/DragSensorNode.h>

namespace CyberX3D {

class CylinderSensorNode : public DragSensorNode {

	SFFloat *diskAngleField;
	SFFloat *minAngleField;
	SFFloat *maxAngleField;
	SFFloat *offsetField;
	SFRotation *rotationField;

public:

	CylinderSensorNode();
	virtual ~CylinderSensorNode();

	////////////////////////////////////////////////
	//	DiskAngle
	////////////////////////////////////////////////

	SFFloat *getDiskAngleField() const;
	
	void setDiskAngle(float value);
	float getDiskAngle() const;

	////////////////////////////////////////////////
	//	MinAngle
	////////////////////////////////////////////////

	SFFloat *getMinAngleField() const;
	
	void setMinAngle(float value);
	float getMinAngle() const;

	////////////////////////////////////////////////
	//	MaxAngle
	////////////////////////////////////////////////

	SFFloat *getMaxAngleField() const;
	
	void setMaxAngle(float value);
	float getMaxAngle() const;

	////////////////////////////////////////////////
	//	Offset
	////////////////////////////////////////////////

	SFFloat *getOffsetField() const;
	
	void setOffset(float value);
	float getOffset() const;

	////////////////////////////////////////////////
	//	Rotation
	////////////////////////////////////////////////

	SFRotation *getRotationChangedField() const;
	
	void setRotationChanged(float value[]);
	void setRotationChanged(float x, float y, float z, float rot);
	void getRotationChanged(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	CylinderSensorNode *next() const;
	CylinderSensorNode *nextTraversal() const;

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

