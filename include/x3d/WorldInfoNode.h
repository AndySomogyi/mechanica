/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	WorldInfoNode.h
*
******************************************************************/

#ifndef _CX3D_WORLDINFONODE_H_
#define _CX3D_WORLDINFONODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/InfoNode.h>

namespace CyberX3D {

class WorldInfoNode : public InfoNode {

	SFString *titleField;
	MFString *infoField;

public:

	WorldInfoNode();
	virtual ~WorldInfoNode();

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	WorldInfoNode *next() const;
	WorldInfoNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	Title
	////////////////////////////////////////////////

	SFString *getTitleField() const;
	
	void setTitle(const char *value);
	const char *getTitle() const;

	////////////////////////////////////////////////
	// Info
	////////////////////////////////////////////////

	MFString *getInfoField() const;

	void addInfo(const char *value);
	int getNInfos() const;
	const char *getInfo(int index) const;

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

	void outputContext(std::ostream& printStream, const char *indentString) const;
};

}

#endif
