/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanSequencer.h
*
******************************************************************/

#ifndef _CX3D_BOOLEANSEQUENCERNODE_H_
#define _CX3D_BOOLEANSEQUENCERNODE_H_

#include <x3d/X3DFields.h>
#include <x3d/SequencerNode.h>

namespace CyberX3D {

class BooleanSequencerNode : public SequencerNode {

	MFBool *keyValueField;
	SFBool *valueField;

public:

	BooleanSequencerNode();
	virtual ~BooleanSequencerNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFBool *getKeyValueField() const;

	void addKeyValue(bool value);
	int getNKeyValues() const;
	bool getKeyValue(int index) const;
	
	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFBool *getValueField() const;

	void setValue(bool vector);
	bool getValue() const;

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

	BooleanSequencerNode *next() const;
	BooleanSequencerNode *nextTraversal() const;
};

}

#endif
