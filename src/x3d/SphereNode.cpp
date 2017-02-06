/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	SphereNode.cpp
*
******************************************************************/

#include <x3d/SphereNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

SphereNode::SphereNode() 
{
	setHeaderFlag(false);
	setType(SPHERE_NODE);

	///////////////////////////
	// Exposed Field 
	///////////////////////////

	// radius exposed field
	radiusField = new SFFloat(1.0f);
	addExposedField(radiusFieldString, radiusField);

	///////////////////////////
	// Slice
	///////////////////////////

	setSlices(DEFAULT_SPHERE_SLICES);
}

SphereNode::~SphereNode() 
{
}

////////////////////////////////////////////////
//	Radius
////////////////////////////////////////////////

SFFloat *SphereNode::getRadiusField() const
{
	if (isInstanceNode() == false)
		return radiusField;
	return (SFFloat *)getExposedField(radiusFieldString);
}
	
void SphereNode::setRadius(float value) 
{
	getRadiusField()->setValue(value);
}

float SphereNode::getRadius() const
{
	return getRadiusField()->getValue();
} 

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

SphereNode *SphereNode::next() const
{
	return (SphereNode *)Node::next(getType());
}

SphereNode *SphereNode::nextTraversal() const
{
	return (SphereNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool SphereNode::isChildNodeType(Node *node) const
{
	return false;
}

void SphereNode::initialize() 
{
	recomputeBoundingBox();
#ifdef CX3D_SUPPORT_OPENGL
		recomputeDisplayList();
#endif
}

void SphereNode::uninitialize() 
{
}

void SphereNode::update() 
{
}

////////////////////////////////////////////////
//	BoundingBox
////////////////////////////////////////////////

void SphereNode::recomputeBoundingBox() 
{
	setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
	setBoundingBoxSize(getRadius(), getRadius(), getRadius());
}

////////////////////////////////////////////////
//	Polygons
////////////////////////////////////////////////

int SphereNode::getNPolygons() const
{
	int slices = getSlices();

	return (slices * slices);
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void SphereNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	printStream << indentString << "\t" << "radius " << getRadius() << std::endl;
}

////////////////////////////////////////////////
//	SphereNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void SphereNode::recomputeDisplayList() 
{
	unsigned int nCurrentDisplayList = getDisplayList();
	if (0 < nCurrentDisplayList)
		glDeleteLists(nCurrentDisplayList, 1);

	int slices = getSlices();

	unsigned int nNewDisplayList = glGenLists(1);
	glNewList(nNewDisplayList, GL_COMPILE);
		glFrontFace(GL_CCW);

	    glPushMatrix ();
	
		glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
	    glRotatef (180.0, 0.0, 1.0, 0.0);
		
		glMatrixMode(GL_MODELVIEW);

	    glRotatef (90.0, 1.0, 0.0, 0.0);
	    glRotatef (180.0, 0.0, 0.0, 1.0);

	    GLUquadricObj *quadObj = gluNewQuadric ();
	    gluQuadricDrawStyle(quadObj, GLU_FILL);
	    gluQuadricNormals(quadObj, GLU_SMOOTH);
	    gluQuadricTexture(quadObj, GL_TRUE);
	    gluSphere(quadObj, getRadius(), slices, slices);
		gluDeleteQuadric(quadObj);

	    glPopMatrix ();

	glEndList();

	setDisplayList(nNewDisplayList);
};

#endif
