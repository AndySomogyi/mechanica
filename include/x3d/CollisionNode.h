/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CollisionNode.h
*
******************************************************************/

#ifndef _CX3D_COLLISIONNODE_H_
#define _CX3D_COLLISIONNODE_H_

#include <x3d/BoundedGroupingNode.h>

namespace CyberX3D {

class CollisionNode : public BoundedGroupingNode {

	SFBool *collideField;
	SFTime *collideTimeField;

public:

	CollisionNode();
	virtual ~CollisionNode();

	////////////////////////////////////////////////
	//	collide
	////////////////////////////////////////////////

	SFBool *getCollideField() const;

	void setCollide(bool  value);
	void setCollide(int value);
	bool getCollide() const;

	////////////////////////////////////////////////
	//	collideTime
	////////////////////////////////////////////////

	SFTime *getCollideTimeField() const;

	void setCollideTime(double value);
	double getCollideTime() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	CollisionNode *next() const;
	CollisionNode *nextTraversal() const;

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

