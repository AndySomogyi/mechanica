/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	XMLNode.h
*
******************************************************************/

#ifndef _CX3D_XMLNODE_H_
#define _CX3D_XMLNODE_H_

#include <x3d/Node.h>
#include <x3d/XMLElement.h>

namespace CyberX3D {

class XMLNode : public Node {

public:

	XMLNode();
	virtual ~XMLNode();

	////////////////////////////////////////////////
	//	Element Field
	////////////////////////////////////////////////

	XMLElement *getElement(const char *eleName) const;
	int getNElements() const;
	void addElement(XMLElement *ele);
	void addElement(const char *name, XMLElement *ele);
	void addElement(const char *name, const char *value);
	XMLElement *getElement(int index) const;
/*
	bool removeElement(XMLElement *ele);
	bool removeElement(const char *eleName);
	int getElementNumber(XMLElement ele);
*/

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	XMLNode *next() const;
	XMLNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	virtual functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();
	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif
