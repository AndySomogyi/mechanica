/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	RouteList.cpp
*
******************************************************************/

#include <x3d/RouteList.h>

using namespace CyberX3D;

RouteList::RouteList()
{
}

RouteList::~RouteList()
{
}

void RouteList::addRoute(Route *route) 
{
	addNode(route);
}

Route *RouteList::getRoutes() const
{
	return (Route *)getNodes();
}

Route *RouteList::getRoute(int n)
{
	return (Route *)getNode(n);
}

int RouteList::getNRoutes()
{
	return getNNodes();
}
