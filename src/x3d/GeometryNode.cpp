/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	GeometryNode.cpp
*
******************************************************************/

#include <x3d/GeometryNode.h>
#include <x3d/Graphic3D.h>
#include <OpenGL/gl.h>

using namespace CyberX3D;

GeometryNode::GeometryNode() 
{

	// display list field
	dispListField = new SFInt32(0);
	dispListField->setName(displayListPrivateFieldString);
	addPrivateField(dispListField);

	setDisplayList(0);

}

GeometryNode::~GeometryNode() 
{
}

////////////////////////////////////////////////
//	OpenGL
////////////////////////////////////////////////



SFInt32 *GeometryNode::getDisplayListField() const
{
	if (isInstanceNode() == false)
		return dispListField;
	return (SFInt32 *)getPrivateField(displayListPrivateFieldString);
}

void GeometryNode::setDisplayList(unsigned int n) 
{
	getDisplayListField()->setValue((int)n);
}

unsigned int GeometryNode::getDisplayList() const
{
	return (unsigned int)getDisplayListField()->getValue();
} 

void GeometryNode::draw() const
{
	unsigned int nDisplayList = getDisplayList();
	if (0 < nDisplayList)
		glCallList(nDisplayList);
}

