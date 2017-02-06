/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanToggleNode.h
*
******************************************************************/

#ifndef _CX3D_BOOLEANTOGGLENODE_H_
#define _CX3D_BOOLEANTOGGLENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class BooleanToggleNode : public Node {
	
	SFBool *set_booleanField;
	SFBool *toggleField;

public:

	BooleanToggleNode();
	virtual ~BooleanToggleNode();

	////////////////////////////////////////////////
	//	SetBooleanEvent
	////////////////////////////////////////////////

	SFBool *getBooleanField() const;
	
	void setBoolean(bool value);
	bool getBoolean() const;
	bool isBoolean() const;

	////////////////////////////////////////////////
	//	IntegerKey
	////////////////////////////////////////////////

	SFBool* getToggleField() const;
	
	void setToggle(bool value);
	bool getToggle() const;
	bool isToggle() const;

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

	BooleanToggleNode *next() const;
	BooleanToggleNode *nextTraversal() const;

};

}

#endif

