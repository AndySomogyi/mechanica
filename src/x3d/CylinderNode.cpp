/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	CylinderNode.cpp
*
******************************************************************/

#include <x3d/CylinderNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

CylinderNode::CylinderNode() 
{
	setHeaderFlag(false);
	setType(CYLINDER_NODE);

	// radius field
	radiusField = new SFFloat(1.0f);
	addExposedField(radiusFieldString, radiusField);

	// height field
	heightField = new SFFloat(2.0f);
	addExposedField(heightFieldString, heightField);

	// top field
	topField = new SFBool(true);
	addExposedField(topFieldString, topField);

	// side field
	sideField = new SFBool(true);
	addExposedField(sideFieldString, sideField);

	// bottom field
	bottomField = new SFBool(true);
	addExposedField(bottomFieldString, bottomField);

	///////////////////////////
	// Slice
	///////////////////////////

	setSlices(DEFAULT_CYLINDERNODE_SLICES);
}

CylinderNode::~CylinderNode() 
{
}

////////////////////////////////////////////////
//	radius
////////////////////////////////////////////////

SFFloat *CylinderNode::getRadiusField() const
{
	if (isInstanceNode() == false)
		return radiusField;
	return (SFFloat *)getExposedField(radiusFieldString);
}

void CylinderNode::setRadius(float value) 
{
	getRadiusField()->setValue(value);
}

float CylinderNode::getRadius()  const
{
	return getRadiusField()->getValue();
}

////////////////////////////////////////////////
//	height
////////////////////////////////////////////////

SFFloat *CylinderNode::getHeightField() const
{
	if (isInstanceNode() == false)
		return heightField;
	return (SFFloat *)getExposedField(heightFieldString);
}

void CylinderNode::setHeight(float value) 
{
	getHeightField()->setValue(value);
}

float CylinderNode::getHeight()  const
{
	return getHeightField()->getValue();
}

////////////////////////////////////////////////
//	top
////////////////////////////////////////////////

SFBool *CylinderNode::getTopField() const
{
	if (isInstanceNode() == false)
		return topField;
	return (SFBool *)getExposedField(topFieldString);
}

void CylinderNode::setTop(bool value) 
{
	getTopField()->setValue(value);
}

void CylinderNode::setTop(int value) 
{
	setTop(value ? true : false);
}

bool CylinderNode::getTop()  const
{
	return getTopField()->getValue();
}

////////////////////////////////////////////////
//	side
////////////////////////////////////////////////

SFBool *CylinderNode::getSideField() const
{
	if (isInstanceNode() == false)
		return sideField;
	return (SFBool *)getExposedField(sideFieldString);
}

void CylinderNode::setSide(bool value) 
{
	getSideField()->setValue(value);
}

void CylinderNode::setSide(int value) 
{
	setSide(value ? true : false);
}

bool CylinderNode::getSide()  const
{
	return getSideField()->getValue();
}

////////////////////////////////////////////////
//	bottom
////////////////////////////////////////////////

SFBool *CylinderNode::getBottomField() const
{
	if (isInstanceNode() == false)
		return bottomField;
	return (SFBool *)getExposedField(bottomFieldString);
}

void CylinderNode::setBottom(bool  value) 
{
	getBottomField()->setValue(value);
}

void CylinderNode::setBottom(int value) 
{
	setBottom(value ? true : false);
}

bool  CylinderNode::getBottom()  const
{
	return getBottomField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

CylinderNode *CylinderNode::next()  const
{
	return (CylinderNode *)Node::next(getType());
}

CylinderNode *CylinderNode::nextTraversal()  const
{
	return (CylinderNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool CylinderNode::isChildNodeType(Node *node) const
{
	return false;
}

void CylinderNode::initialize() 
{
	recomputeBoundingBox();
#ifdef CX3D_SUPPORT_OPENGL
	recomputeDisplayList();
#endif
}

void CylinderNode::uninitialize() 
{
}

void CylinderNode::update() 
{
}

////////////////////////////////////////////////
//	BoundingBox
////////////////////////////////////////////////

void CylinderNode::recomputeBoundingBox() 
{
	setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
	setBoundingBoxSize(getRadius(), getHeight()/2.0f, getRadius());
}

////////////////////////////////////////////////
//	Polygons
////////////////////////////////////////////////

int CylinderNode::getNPolygons() const
{
	int nPolys = 0;
	int slices = getSlices();
	
	if (getTop() == true)
		nPolys += slices;

	if (getSide() == true)
		nPolys += slices;

	if (getBottom() == true)
		nPolys += slices;

	return nPolys;
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void CylinderNode::outputContext(std::ostream &printStream, const char *indentString)  const
{
	SFBool *top = getTopField();
	SFBool *side = getSideField();
	SFBool *bottom = getBottomField();

	printStream << indentString << "\t" << "radius " << getRadius() << std::endl;
	printStream << indentString << "\t" << "height " << getHeight() << std::endl;
	printStream << indentString << "\t" << "side " << side << std::endl;
	printStream << indentString << "\t" << "top " << top << std::endl;
	printStream << indentString << "\t" << "bottom " << bottom << std::endl;
}

////////////////////////////////////////////////
//	CylinderNode::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void CylinderNode::recomputeDisplayList() 
{
	unsigned int nCurrentDisplayList = getDisplayList();
	if (0 < nCurrentDisplayList)
		glDeleteLists(nCurrentDisplayList, 1);

	int slices = getSlices();

	unsigned int nNewDisplayList = glGenLists(1);
	glNewList(nNewDisplayList, GL_COMPILE);
		GLUquadricObj *quadObj;

		glFrontFace(GL_CCW);

	    glPushMatrix ();

		glMatrixMode(GL_TEXTURE);
		glLoadIdentity();
	    glRotatef (180.0, 0.0, 1.0, 0.0);

		glMatrixMode(GL_MODELVIEW);

	    glRotatef (180.0, 0.0, 1.0, 0.0);
	    glRotatef (90.0, 1.0, 0.0, 0.0);
	    glTranslatef (0.0, 0.0, -getHeight()/2.0f);

		if (getSide()) {
		    quadObj = gluNewQuadric ();
		    gluQuadricDrawStyle(quadObj, GLU_FILL);
		    gluQuadricNormals(quadObj, GLU_SMOOTH);
		    gluQuadricTexture(quadObj, GL_TRUE);
		    gluCylinder(quadObj, getRadius(), getRadius(), getHeight(), slices, 2);
			gluDeleteQuadric(quadObj);
		}

		if (getTop()) {
		    glPushMatrix ();
		    glRotatef (180.0, 1.0, 0.0, 0.0);
		    quadObj = gluNewQuadric ();
		    gluQuadricTexture(quadObj, GL_TRUE);
			gluDisk(quadObj, 0.0, getRadius(), slices, 2);
			gluDeleteQuadric(quadObj);
		    glPopMatrix ();
		}

		if (getBottom()) {
		    glTranslatef (0.0, 0.0, getHeight());
		    quadObj = gluNewQuadric ();
		    gluQuadricTexture(quadObj, GL_TRUE);
			gluDisk(quadObj, 0.0, getRadius(), slices, 2);
			gluDeleteQuadric(quadObj);
		}

	    glPopMatrix ();
	glEndList();

	setDisplayList(nNewDisplayList);
};

#endif
