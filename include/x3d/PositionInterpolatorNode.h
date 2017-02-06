/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PositionInterpolatorNode.h
*
******************************************************************/

#ifndef _CX3D_POSITIONINTERPOLATOR_H_
#define _CX3D_POSITIONINTERPOLATOR_H_

#include <x3d/InterpolatorNode.h>

namespace CyberX3D {

class PositionInterpolatorNode : public InterpolatorNode {

	MFVec3f *keyValueField;
	SFVec3f *valueField;

public:

	PositionInterpolatorNode();
	virtual ~PositionInterpolatorNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFVec3f *getKeyValueField() const;

	void addKeyValue(float vector[]);
	int getNKeyValues() const;
	void getKeyValue(int index, float vector[]) const;

	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFVec3f *getValueField() const;

	void setValue(float vector[]);
	void getValue(float vector[]) const;

	////////////////////////////////////////////////
	//	Virtual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	PositionInterpolatorNode *next() const;
	PositionInterpolatorNode *nextTraversal() const;
};

}

#endif

