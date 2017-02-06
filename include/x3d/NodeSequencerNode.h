/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NodeSequencer.h
*
******************************************************************/

#ifndef _CX3D_NODESEQUENCERNODE_H_
#define _CX3D_NODESEQUENCERNODE_H_

#include <x3d/X3DFields.h>
#include <x3d/SequencerNode.h>

namespace CyberX3D {

class NodeSequencerNode : public SequencerNode {

	MFNode *keyValueField;
	SFNode *valueField;

public:

	NodeSequencerNode();
	virtual ~NodeSequencerNode();

	////////////////////////////////////////////////
	//	keyValue
	////////////////////////////////////////////////
	
	MFNode *getKeyValueField() const;

	void addKeyValue(Node *value);
	int getNKeyValues() const;
	Node *getKeyValue(int index) const;

	////////////////////////////////////////////////
	//	value
	////////////////////////////////////////////////
	
	SFNode *getValueField() const;

	void setValue(Node *value);
	Node *getValue() const;

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

	NodeSequencerNode *next() const;
	NodeSequencerNode *nextTraversal() const;
};

}

#endif
