/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GeometryNode.h
*
******************************************************************/

#ifndef _CX3D_GEOMETRYNODE_H_
#define _CX3D_GEOMETRYNODE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {


const char displayListPrivateFieldString[] = "oglDisplayList";


class GeometryNode : public Node 
{


	SFInt32 *dispListField;


public:

	GeometryNode();
	virtual ~GeometryNode();

	////////////////////////////////////////////////
	//	DisplayList
	////////////////////////////////////////////////



	SFInt32 *getDisplayListField() const;
	void setDisplayList(unsigned int n);
	unsigned int getDisplayList() const;
	virtual void draw() const;


};

}

#endif
