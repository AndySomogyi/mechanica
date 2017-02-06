/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2004
*
*	File:	Scene.cpp
*
*	03/12/04
*		- Added findLastNode();
*	03/17/04
*		- Added findDEFNode() but not implemented yet.
*
******************************************************************/

#include <x3d/Scene.h>
#include <x3d/UrlFile.h>

using namespace CyberX3D;

Scene::Scene() 
{
}

Scene::~Scene() 
{
}

////////////////////////////////////////////////
//	Parse Action
////////////////////////////////////////////////

void Scene::addNode(Node *node, bool initialize) 
{
	moveNode(node);
	if (initialize)
		node->initialize();
}

void Scene::addNodeAtFirst(Node *node, bool initialize) 
{
	moveNodeAtFirst(node);
	if (initialize)
		node->initialize();
}

void Scene::moveNode(Node *node) 
{
	mNodeList.addNode(node);
	node->setParentNode(NULL);
	node->setSceneGraph((SceneGraph *)this);
}

void Scene::moveNodeAtFirst(Node *node) 
{
	mNodeList.addNodeAtFirst(node);
	node->setParentNode(NULL);
	node->setSceneGraph((SceneGraph *)this);
}


////////////////////////////////////////////////
//	find node
////////////////////////////////////////////////

Node *Scene::findNode(const int type) const
{
	Node *node = getRootNode()->nextTraversalByType(type);
	if (node) {
		while (node->isInstanceNode() == true)
			node = node->getReferenceNode();
	}
	return node;
}

Node *Scene::findNode(const char *name) const
{
	if (!name)
		return NULL;
	if (strlen(name) <= 0)
		return NULL;
	Node *node = getRootNode()->nextTraversalByName(name);
	if (node) {
		while (node->isInstanceNode() == true)
			node = node->getReferenceNode();
	}
	return node;
}

Node *Scene::findNode(const int type, const char *name) const
{
	if (name == NULL)
		return NULL;
	if (strlen(name) <= 0)
		return NULL;
	std::string nameStr = name;
	for (Node *node = findNode(type); node != NULL; node = node->nextTraversalSameType()) {
		const char *nodeName = node->getName();
		if (nodeName != NULL) {
			if (nameStr.compare(nodeName) == 0)
				return node;
		}
	}
	return NULL;
}

bool Scene::hasNode(Node *targetNode) const
{
	for (Node *node = getNodes(); node; node = node->nextTraversal()) {
		if (node == targetNode)
			return true;
	}
	return false;
}

Node *Scene::findLastNode(const char *name) const 
{
	if (!name)
		return NULL;
	if (strlen(name) <= 0)
		return NULL;

	Node *findNode = NULL;
	String nameStr(name);
	for (Node *node = getRootNode()->nextTraversal(); node != NULL; node = node->nextTraversal()) {
		if (node->getName() != NULL) {
			if (nameStr.compareTo(node->getName()) == 0)
				findNode = node;
		}
	}
	
	if (findNode) {
		while (findNode->isInstanceNode() == true)
			findNode = findNode->getReferenceNode();
	}
	return findNode;
}

Node *Scene::findDEFNode(const char *name) const
{
	return findNode(name);
}

///////////////////////////////////////////////
//	ROUTE
///////////////////////////////////////////////

Route *Scene::getRoutes() const
{
	return (Route *)mRouteList.getNodes();
}

Route *Scene::getRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField) const
{
	for (Route *route=getRoutes(); route; route=route->next()) {
		if (eventOutNode == route->getEventOutNode() && eventOutField == route->getEventOutField() &&
			eventInNode == route->getEventInNode() && eventInField == route->getEventInField() ) {
			return route;
		}
	}
	return NULL;
}

void Scene::addRoute(Route *route) 
{
	if (route->getEventOutNode() == route->getEventInNode())
		return;
	if (getRoute(route->getEventOutNode(), route->getEventOutField(), route->getEventInNode(), route->getEventInField()))
		return;
	mRouteList.addNode(route);
}

Route *Scene::addRoute(const char *eventOutNodeName, const char *eventOutFieldName, const char *eventInNodeName, const char *eventInFieldName)
{
	Node *eventInNode = findNode(eventInNodeName);
	Node *eventOutNode = findNode(eventOutNodeName);

	Field *eventOutField = NULL;

	if (eventOutNode) {
		eventOutField = eventOutNode->getEventOut(eventOutFieldName);
		if (!eventOutField)
			eventOutField = eventOutNode->getExposedField(eventOutFieldName);
	}

	Field *eventInField = NULL;

	if (eventInNode) {
		eventInField = eventInNode->getEventIn(eventInFieldName);
		if (!eventInField)
			eventInField = eventInNode->getExposedField(eventInFieldName);
	}
	
	if (!eventInNode || !eventOutNode || !eventInField || !eventOutField)
		return NULL;
	Route *route = new Route(eventOutNode, eventOutField, eventInNode, eventInField);
	addRoute(route);
	return route;
}

Route *Scene::addRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField)
{
	Route *route = new Route(eventOutNode, eventOutField, eventInNode, eventInField);
	addRoute(route);
	return route;
}

void Scene::deleteRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField)
{
	Route *route  = getRoute(eventOutNode, eventOutField, eventInNode, eventInField);
	if (route)
		delete route;
}

void Scene::deleteRoutes(Node *node) {
	Route *route = getRoutes();
	while (route) {
		Route *nextRoute = route->next();
		if (node == route->getEventInNode() || node == route->getEventOutNode())
			delete route;
		route = nextRoute;
	}
}

void Scene::deleteEventInFieldRoutes(Node *node, Field *field)
{
	Route	*route = getRoutes();
	while (route) {
		Route *nextRoute = route->next();
		if (route->getEventInNode() == node && route->getEventInField() == field)
			delete route;
		route = nextRoute;
	}
}

void Scene::deleteEventOutFieldRoutes(Node *node, Field *field)
{
	Route	*route = getRoutes();
	while (route) {
		Route *nextRoute = route->next();
		if (route->getEventOutNode() == node && route->getEventOutField() == field)
			delete route;
		route = nextRoute;
	}
}

void Scene::deleteRoutes(Node *node, Field *field)
{
	deleteEventInFieldRoutes(node, field);
	deleteEventOutFieldRoutes(node, field);
}

void Scene::deleteRoute(Route *deleteRoute)
{
	for (Route *route=getRoutes(); route; route=route->next()) {
		if (deleteRoute == route) {
			delete route;
			return;
		}
	}
}

void Scene::removeRoute(Node *eventOutNode, Field *eventOutField, Node *eventInNode, Field *eventInField)
{
	Route *route  = getRoute(eventOutNode, eventOutField, eventInNode, eventInField);
	if (route)
		route->remove();
}

void Scene::removeRoutes(Node *node) 
{
	Route *route = getRoutes();
	while (route) {
		Route *nextRoute = route->next();
		if (node == route->getEventInNode() || node == route->getEventOutNode())
			route->remove();
		route = nextRoute;
	}
}

void Scene::removeEventInFieldRoutes(Node *node, Field *field)
{
	Route	*route = getRoutes();
	while (route) {
		Route *nextRoute = route->next();
		if (route->getEventInNode() == node && route->getEventInField() == field)
			route->remove();
		route = nextRoute;
	}
}

void Scene::removeEventOutFieldRoutes(Node *node, Field *field)
{
	Route	*route = getRoutes();
	while (route) {
		Route *nextRoute = route->next();
		if (route->getEventOutNode() == node && route->getEventOutField() == field)
			route->remove();
		route = nextRoute;
	}
}

void Scene::removeRoutes(Node *node, Field *field)
{
	removeEventInFieldRoutes(node, field);
	removeEventOutFieldRoutes(node, field);
}

void Scene::removeRoute(Route *removeRoute)
{
	for (Route *route=getRoutes(); route; route=route->next()) {
		if (removeRoute == route) {
			route->remove();
			return;
		}
	}
}
