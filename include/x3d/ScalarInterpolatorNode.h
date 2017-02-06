/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ScalarInterpolatorNode.h
*
******************************************************************/

#ifndef _CX3D_SCALARINTERPOLATOR_H_
#define _CX3D_SCALARINTERPOLATOR_H_

#include <x3d/InterpolatorNode.h>

namespace CyberX3D {

class ScalarInterpolatorNode : public InterpolatorNode {

	MFFloat *keyValueField;
	SFFloat *valueField;

public:

	ScalarInterpolatorNode();
	virtual ~ScalarInterpolatorNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFFloat *getKeyValueField() const;

	void addKeyValue(float value);
	int getNKeyValues() const;
	float getKeyValue(int index) const;

	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFFloat *getValueField() const;

	void setValue(float vector);
	float getValue() const;

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

	ScalarInterpolatorNode *next() const;
	ScalarInterpolatorNode *nextTraversal() const;
};

}

#endif
