/******************************************************************
*
*	VRML library for C++
*
*	Copyright (C) Satoshi Konno 1996-1997
*
*	File:	DEFNode.h
*
******************************************************************/

#ifndef _DEFNODE_H_
#define _DEFNODE_H_

#include <x3d/Node.h>

namespace CyberX3D {

class DEFNode : public Node {

public:

	DEFNode();
	virtual ~DEFNode();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	DEFNode *next() const;
	DEFNode *nextTraversal() const;

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

