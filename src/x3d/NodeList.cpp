/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	NodeList.cpp
*
******************************************************************/

#include <x3d/NodeList.h>

using namespace CyberX3D;

NodeList::NodeList() 
{
	RootNode *rootNode = new RootNode();
	setRootNode(rootNode);
}

NodeList::~NodeList() 
{
}
