/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	StaticGroupNode.h
*
******************************************************************/

#ifndef _CX3D_STATICGROUPNODE_H_
#define _CX3D_STATICGROUPNODE_H_

#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {

class StaticGroupNode : public BoundedGroupingNode 
{

public:

	StaticGroupNode();
	virtual ~StaticGroupNode();

	////////////////////////////////////////////////
	//	Output
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	StaticGroupNode *next() const;
	StaticGroupNode *nextTraversal() const;

};

}

#endif
