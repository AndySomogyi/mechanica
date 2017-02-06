/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ColorInterpolatorNode.h
*
******************************************************************/

#ifndef _CX3D_COLORINTERPOLATOR_H_
#define _CX3D_COLORINTERPOLATOR_H_

#include <x3d/InterpolatorNode.h>

namespace CyberX3D {

class ColorInterpolatorNode : public InterpolatorNode {

	MFColor *keyValueField;
	SFColor *valueField;

public:

	ColorInterpolatorNode();
	virtual ~ColorInterpolatorNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFColor *getKeyValueField() const;

	void addKeyValue(float color[]);
	int getNKeyValues() const;
	void getKeyValue(int index, float color[]) const;

	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFColor *getValueField() const;

	void setValue(float color[]);
	void getValue(float color[]) const;

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

	ColorInterpolatorNode *next() const;
	ColorInterpolatorNode *nextTraversal() const;
};

}

#endif
