/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NormalInterpolatorNode.h
*
******************************************************************/

#ifndef _CX3D_NORMALINTERPOLATOR_H_
#define _CX3D_NORMALINTERPOLATOR_H_

#include <x3d/InterpolatorNode.h>

namespace CyberX3D {

class NormalInterpolatorNode : public InterpolatorNode {

	MFVec3f *keyValueField;
	SFVec3f *valueField;

public:

	NormalInterpolatorNode();
	virtual ~NormalInterpolatorNode();

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

	NormalInterpolatorNode *next() const;
	NormalInterpolatorNode *nextTraversal() const;

};

}

#endif

