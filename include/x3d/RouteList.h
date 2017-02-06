/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	RouteList.h
*
******************************************************************/

#ifndef _CX3D_ROUTELIST_H_
#define _CX3D_ROUTELIST_H_

#include <x3d/LinkedList.h>
#include <x3d/Route.h>

namespace CyberX3D {

class RouteList : public LinkedList<Route> {

public:

	RouteList();
	virtual ~RouteList();

	void addRoute(Route *route);
	Route *getRoutes() const;
	Route *getRoute(int n);
	int getNRoutes();
};

}

#endif
