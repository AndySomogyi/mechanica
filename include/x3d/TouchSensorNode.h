/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TouchSensorNode.h
*
******************************************************************/

#ifndef _CX3D_TOUCHSENSOR_H_
#define _CX3D_TOUCHSENSOR_H_

#include <x3d/SensorNode.h>

namespace CyberX3D {

class TouchSensorNode : public SensorNode {

	SFVec3f *hitNormalField;
	SFVec2f *hitTexCoordField;
	SFVec3f *hitPointField;
	SFBool *isOverField;
	SFTime *touchTimeField;
	
public:

	TouchSensorNode();
	virtual ~TouchSensorNode();

	////////////////////////////////////////////////
	//	isOver
	////////////////////////////////////////////////
	
	SFBool *getIsOverField() const;

	void setIsOver(bool  value);
	void setIsOver(int value);
	bool  getIsOver() const;
	bool  isOver() const;

	////////////////////////////////////////////////
	//	hitNormal
	////////////////////////////////////////////////
	
	SFVec3f *getHitNormalChangedField() const;

	void setHitNormalChanged(float value[]);
	void setHitNormalChanged(float x, float y, float z);
	void getHitNormalChanged(float value[]) const;

	////////////////////////////////////////////////
	//	hitPoint
	////////////////////////////////////////////////
	
	SFVec3f *getHitPointChangedField() const;

	void setHitPointChanged(float value[]);
	void setHitPointChanged(float x, float y, float z);
	void getHitPointChanged(float value[]) const;

	////////////////////////////////////////////////
	//	hitTexCoord
	////////////////////////////////////////////////
	
	SFVec2f *getHitTexCoordField() const;

	void setHitTexCoord(float value[]);
	void setHitTexCoord(float x, float y);
	void getHitTexCoord(float value[]) const;

	////////////////////////////////////////////////
	//	ExitTime
	////////////////////////////////////////////////
	
	SFTime *getTouchTimeField() const;

	void setTouchTime(double value);
	double getTouchTime() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TouchSensorNode *next() const;
	TouchSensorNode *nextTraversal() const;

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
