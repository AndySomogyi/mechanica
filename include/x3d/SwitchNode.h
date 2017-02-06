/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SwitchNode.h
*
******************************************************************/

#ifndef _CX3D_SWITCHNODE_H_
#define _CX3D_SWITCHNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {

class SwitchNode : public BoundedGroupingNode {

	SFInt32 *whichChoiceField;

public:

	SwitchNode();
	virtual ~SwitchNode();

	////////////////////////////////////////////////
	//	whichChoice
	////////////////////////////////////////////////

	SFInt32 *getWhichChoiceField() const;

	void setWhichChoice(int value);
	int getWhichChoice() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	SwitchNode *next() const;
	SwitchNode *nextTraversal() const;

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

