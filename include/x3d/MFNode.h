/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MFNode.h
*
******************************************************************/

#ifndef _CX3D_MFNODE_H_
#define _CX3D_MFNODE_H_

#include <x3d/MField.h>
#include <x3d/SFNode.h>

namespace CyberX3D {

class MFNode : public MField {

	static	int	mInit;

public:

	MFNode();

	void addValue(Node *value);
	void addValue(SFNode *sfvalue);
	void addValue(const char *value);

	void insertValue(int index, Node *value);
	Node *get1Value(int index) const;
	void set1Value(int index, Node *value);

	void setValue(MField *mfield);
	void setValue(MFNode *values);
	void setValue(int size, Node *values[]);

	int getValueCount() const 
	{
		return 1;
	}

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream& printStream, const char *indentString) const;

};

}

#endif
