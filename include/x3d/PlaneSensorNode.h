/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PlaneSensorNode.h
*
******************************************************************/

#ifndef _CX3D_PLANESENSOR_H_
#define _CX3D_PLANESENSOR_H_

#include <x3d/DragSensorNode.h>

namespace CyberX3D {

class PlaneSensorNode : public DragSensorNode {

	SFVec2f *minPositionField;
	SFVec2f *maxPositionField;
	SFVec3f *offsetField;
	SFVec3f *translationField;
	
public:

	PlaneSensorNode();
	virtual ~PlaneSensorNode();

	////////////////////////////////////////////////
	//	MinPosition
	////////////////////////////////////////////////
	
	SFVec2f *getMinPositionField() const;

	void setMinPosition(float value[]);
	void setMinPosition(float x, float y);
	void getMinPosition(float value[]) const;
	void getMinPosition(float *x, float *y) const;

	////////////////////////////////////////////////
	//	MaxPosition
	////////////////////////////////////////////////
	
	SFVec2f *getMaxPositionField() const;

	void setMaxPosition(float value[]);
	void setMaxPosition(float x, float y);
	void getMaxPosition(float value[]) const;
	void getMaxPosition(float *x, float *y) const;

	////////////////////////////////////////////////
	//	Offset
	////////////////////////////////////////////////
	
	SFVec3f *getOffsetField() const;

	void setOffset(float value[]);
	void getOffset(float value[]) const;

	////////////////////////////////////////////////
	//	Translation
	////////////////////////////////////////////////
	
	SFVec3f *getTranslationChangedField() const;

	void setTranslationChanged(float value[]);
	void setTranslationChanged(float x, float y, float z);
	void getTranslationChanged(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	PlaneSensorNode *next() const;
	PlaneSensorNode *nextTraversal() const;

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

