/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CoordinateInterpolator2DNode.h
*
******************************************************************/

#ifndef _CX3D_COORDINATEINTERPOLATOR2D_H_
#define _CX3D_COORDINATEINTERPOLATOR2D_H_

#include <x3d/X3DFields.h>
#include <x3d/InterpolatorNode.h>

namespace CyberX3D {

class CoordinateInterpolator2DNode : public InterpolatorNode {

	MFVec2f *keyValueField;
	SFVec2f *valueField;

public:

	CoordinateInterpolator2DNode();
	virtual ~CoordinateInterpolator2DNode();

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
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	CoordinateInterpolator2DNode *next() const;
	CoordinateInterpolator2DNode *nextTraversal() const;
};

}

#endif

