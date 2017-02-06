/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GroupingNode.h
*
******************************************************************/

#ifndef _CX3D_GROUPINGNODE_H_
#define _CX3D_GROUPINGNODE_H_

#include <x3d/Node.h>

namespace CyberX3D {

class GroupingNode : public Node {

public:

	GroupingNode();
	virtual ~GroupingNode();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	GroupingNode *next() const;
	GroupingNode *nextTraversal() const;
};

}

#endif

