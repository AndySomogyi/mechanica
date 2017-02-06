/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	PositionInterpolator2DNode.h
*
******************************************************************/

#ifndef _CX3D_POSITIONINTERPOLATOR2D_H_
#define _CX3D_POSITIONINTERPOLATOR2D_H_

#include <x3d/InterpolatorNode.h>

namespace CyberX3D {

class PositionInterpolator2DNode : public InterpolatorNode {

	MFVec2f *keyValueField;
	SFVec2f *valueField;

public:

	PositionInterpolator2DNode();
	virtual ~PositionInterpolator2DNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFVec2f *getKeyValueField() const;

	void addKeyValue(float vector[]);
	int getNKeyValues() const;
	void getKeyValue(int index, float vector[]) const;

	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFVec2f *getValueField() const;

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

	PositionInterpolator2DNode *next() const;
	PositionInterpolator2DNode *nextTraversal() const;
};

}

#endif

