/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SphereSensorNode.h
*
******************************************************************/

#ifndef _CX3D_SPHERESENSOR_H_
#define _CX3D_SPHERESENSOR_H_

#include <x3d/DragSensorNode.h>

namespace CyberX3D {

class SphereSensorNode : public DragSensorNode {

	SFRotation *offsetField;
	SFRotation *rotationField;
	
public:

	SphereSensorNode();
	virtual ~SphereSensorNode();

	////////////////////////////////////////////////
	//	Offset
	////////////////////////////////////////////////

	SFRotation *getOffsetField() const;
	
	void setOffset(float value[]);
	void getOffset(float value[]) const;

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

	SphereSensorNode *next() const;
	SphereSensorNode *nextTraversal() const;

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

