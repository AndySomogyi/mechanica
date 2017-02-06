/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NodeList.h
*
******************************************************************/

#ifndef _CX3D_NODELIST_H_
#define _CX3D_NODELIST_H_

#include <x3d/LinkedList.h>
#include <x3d/RootNode.h>

namespace CyberX3D {

class NodeList : public LinkedList<Node> {

public:

	NodeList();
	virtual ~NodeList();
};

}

#endif
