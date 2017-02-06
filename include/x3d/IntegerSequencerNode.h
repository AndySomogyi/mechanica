/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	IntegerSequencer.h
*
******************************************************************/

#ifndef _CX3D_INTEGERSEQUENCERNODE_H_
#define _CX3D_INTEGERSEQUENCERNODE_H_

#include <x3d/SequencerNode.h>
#include <x3d/X3DFields.h>

namespace CyberX3D {

class IntegerSequencerNode : public SequencerNode {

	MFInt32 *keyValueField;
	SFInt32 *valueField;

public:

	IntegerSequencerNode();
	virtual ~IntegerSequencerNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFInt32 *getKeyValueField() const;

	void addKeyValue(int value);
	int getNKeyValues() const;
	int getKeyValue(int index) const;

	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFInt32 *getValueField() const;

	void setValue(int vector);
	int getValue() const;

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

	IntegerSequencerNode *next() const;
	IntegerSequencerNode *nextTraversal() const;
};

}

#endif
