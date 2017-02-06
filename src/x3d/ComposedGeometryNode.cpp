/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	ComposedGeometryNode.cpp
*
*	Revisions:
*
*		11/25/02
*			- The first release.
*
******************************************************************/

#include <x3d/ComposedGeometryNode.h>

using namespace CyberX3D;

static const char colorExposedFieldString[] = "color";
static const char coordExposedFieldString[] = "coord";
static const char normalExposedFieldString[] = "normal";
static const char texCoordExposedFieldString[] = "texCoord";

ComposedGeometryNode::ComposedGeometryNode() 
{
	///////////////////////////
	// ExposedField 
	///////////////////////////

	// color field
	colorField = new SFNode();
	addExposedField(colorExposedFieldString, colorField);

	// coord field
	coordField = new SFNode();
	addExposedField(coordExposedFieldString, coordField);

	// normal field
	normalField = new SFNode();
	addExposedField(normalExposedFieldString, normalField);

	// texCoord field
	texCoordField = new SFNode();
	addExposedField(texCoordExposedFieldString, texCoordField);

	///////////////////////////
	// Field 
	///////////////////////////

	// ccw  field
	ccwField = new SFBool(true);
	ccwField->setName(ccwFieldString);
	addField(ccwField);

	// colorPerVertex  field
	colorPerVertexField = new SFBool(true);
	colorPerVertexField->setName(colorPerVertexFieldString);
	addField(colorPerVertexField);

	// normalPerVertex  field
	normalPerVertexField = new SFBool(true);
	normalPerVertexField->setName(normalPerVertexFieldString);
	addField(normalPerVertexField);

	// solid  field
	solidField = new SFBool(true);
	solidField->setName(solidFieldString);
	addField(solidField);
}

ComposedGeometryNode::~ComposedGeometryNode() 
{
}

////////////////////////////////////////////////
//	Color
////////////////////////////////////////////////

SFNode *ComposedGeometryNode::getColorField() const 
{
	if (isInstanceNode() == false)
		return colorField;
	return (SFNode *)getExposedField(colorExposedFieldString);
}
	
////////////////////////////////////////////////
//	Coord
////////////////////////////////////////////////

SFNode *ComposedGeometryNode::getCoordField() const 
{
	if (isInstanceNode() == false)
		return coordField;
	return (SFNode *)getExposedField(coordExposedFieldString);
}
	
////////////////////////////////////////////////
//	Normal
////////////////////////////////////////////////

SFNode *ComposedGeometryNode::getNormalField() const 
{
	if (isInstanceNode() == false)
		return normalField;
	return (SFNode *)getExposedField(normalExposedFieldString);
}
	
////////////////////////////////////////////////
//	texCoord
////////////////////////////////////////////////

SFNode *ComposedGeometryNode::getTexCoordField() const
{
	if (isInstanceNode() == false)
		return texCoordField;
	return (SFNode *)getExposedField(texCoordExposedFieldString);
}
	
////////////////////////////////////////////////
//	CCW
////////////////////////////////////////////////

SFBool *ComposedGeometryNode::getCCWField() const
{
	if (isInstanceNode() == false)
		return ccwField;
	return (SFBool *)getField(ccwFieldString);
}
	
void ComposedGeometryNode::setCCW(bool value) 
{
	getCCWField()->setValue(value);
}

void ComposedGeometryNode::setCCW(int value) 
{
	setCCW(value ? true : false);
}

bool ComposedGeometryNode::getCCW() const 
{
	return getCCWField()->getValue();
}

////////////////////////////////////////////////
//	ColorPerVertex
////////////////////////////////////////////////

SFBool *ComposedGeometryNode::getColorPerVertexField() const
{
	if (isInstanceNode() == false)
		return colorPerVertexField;
	return (SFBool *)getField(colorPerVertexFieldString);
}
	
void ComposedGeometryNode::setColorPerVertex(bool value) 
{
	getColorPerVertexField()->setValue(value);
}

void ComposedGeometryNode::setColorPerVertex(int value) 
{
	setColorPerVertex(value ? true : false);
}

bool ComposedGeometryNode::getColorPerVertex() const 
{
	return getColorPerVertexField()->getValue();
}

////////////////////////////////////////////////
//	NormalPerVertex
////////////////////////////////////////////////

SFBool *ComposedGeometryNode::getNormalPerVertexField() const
{
	if (isInstanceNode() == false)
		return normalPerVertexField;
	return (SFBool *)getField(normalPerVertexFieldString);
}

void ComposedGeometryNode::setNormalPerVertex(bool value) 
{
	getNormalPerVertexField()->setValue(value);
}

void ComposedGeometryNode::setNormalPerVertex(int value) 
{
	setNormalPerVertex(value ? true : false);
}

bool ComposedGeometryNode::getNormalPerVertex() const 
{
	return getNormalPerVertexField()->getValue();
}

////////////////////////////////////////////////
//	Solid
////////////////////////////////////////////////

SFBool *ComposedGeometryNode::getSolidField() const
{
	if (isInstanceNode() == false)
		return solidField;
	return (SFBool *)getField(solidFieldString);
}

void ComposedGeometryNode::setSolid(bool value) 
{
	getSolidField()->setValue(value);
}
	
void ComposedGeometryNode::setSolid(int value) 
{
	setSolid(value ? true : false);
}

bool ComposedGeometryNode::getSolid() const 
{
	return getSolidField()->getValue();
}

////////////////////////////////////////////////////////////
//	TriangleSetNode::recomputeBoundingBox
////////////////////////////////////////////////////////////

void ComposedGeometryNode::recomputeBoundingBox() 
{
	CoordinateNode *coordinateNode = getCoordinateNodes();
	if (!coordinateNode) {
		setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
		setBoundingBoxSize(-1.0f, -1.0f, -1.0f);
		return;
	}

	BoundingBox bbox;
	float		point[3];

	int nPoints = coordinateNode->getNPoints();
	for (int n=0; n<nPoints; n++) {
		coordinateNode->getPoint(n, point);
		bbox.addPoint(point);
	}

	setBoundingBox(&bbox);
}

