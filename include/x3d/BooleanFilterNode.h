/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	BooleanFilterNode.h
*
******************************************************************/

#ifndef _CX3D_BOOLEANFILTERNODE_H_
#define _CX3D_BOOLEANFILTERNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

class BooleanFilterNode : public Node 
{
	SFBool *set_booleanField;
	SFBool *inputFalseField;
	SFBool *inputNegateField;
	SFBool *inputTrueField;

public:

	BooleanFilterNode();
	virtual ~BooleanFilterNode();

	////////////////////////////////////////////////
	//	setBoolean
	////////////////////////////////////////////////

	SFBool *getBooleanField() const;
	
	void setBoolean(bool value);
	bool getBoolean() const;
	bool isBoolean() const;

	////////////////////////////////////////////////
	//	inputFalse
	////////////////////////////////////////////////

	SFBool* getInputFalseField() const;
	
	void setInputFalse(bool value);
	bool getInputFalse() const;
	bool isInputFalse() const;

	////////////////////////////////////////////////
	//	inputNegate
	////////////////////////////////////////////////

	SFBool* getInputNegateField() const;
	
	void setInputNegate(bool value);
	bool getInputNegate() const;
	bool isInputNegate() const;

	////////////////////////////////////////////////
	//	inputTrue
	////////////////////////////////////////////////

	SFBool* getInputTrueField() const;
	
	void setInputTrue(bool value);
	bool getInputTrue() const;
	bool isInputTrue() const;

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

	BooleanFilterNode *next() const;
	BooleanFilterNode *nextTraversal() const;

};

}

#endif

