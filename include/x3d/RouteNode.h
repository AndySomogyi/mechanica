/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	RouteNode.h
*
******************************************************************/

#ifndef _CX3D_ROUTENODE_H_
#define _CX3D_ROUTENODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

//<ROUTE fromField="isActive" fromNode="red_pipe" toField="toggle" toNode="linkFailure"/>

class RouteNode : public Node {
	
	SFString *fromField;
	SFString *fromNode;
	SFString *toField;
	SFString *toNode;

public:

	RouteNode();
	virtual ~RouteNode();

	////////////////////////////////////////////////
	//	Field
	////////////////////////////////////////////////

	SFString *getFromField() const;
	SFString *getFromNode() const;
	SFString *getToField() const;
	SFString *getToNode() const;

	void setFromFieldName(const char *value);
	void setFromNodeName(const char *value);
	void setToFieldName(const char *value);
	void setToNodeName(const char *value);

	const char *getFromFieldName() const;
	const char *getFromNodeName() const;
	const char *getToFieldName() const;
	const char *getToNodeName() const;

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

	RouteNode *next() const;
	RouteNode *nextTraversal() const;
};

}

#endif

