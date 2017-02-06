/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	InlineNode.h
*
******************************************************************/

#ifndef _CX3D_INLINENODE_H_
#define _CX3D_INLINENODE_H_

#include <x3d/BoundedNode.h>

namespace CyberX3D {

class InlineNode : public BoundedNode {

	MFString *urlField;
	SFBool *loadField;

public:

	InlineNode();
	virtual ~InlineNode();

	////////////////////////////////////////////////
	// Url
	////////////////////////////////////////////////

	MFString *getUrlField() const;

	void addUrl(const char *value);
	int getNUrls() const;
	const char *getUrl(int index) const;
	void setUrl(int index, const char *urlString);

	////////////////////////////////////////////////
	//	Load (X3D)
	////////////////////////////////////////////////

	SFBool *getLoadField() const;
	
	void setLoad(bool value);
	bool getLoad() const;
	bool isLoad() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	InlineNode *next() const;
	InlineNode *nextTraversal() const;

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

