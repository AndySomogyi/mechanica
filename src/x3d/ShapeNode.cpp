/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File: ShapeNode.cpp
*
*	Revision:
*
*	12/05/03
*		- Shenyang <shenyang@163.net>
*		- Dennis <dennis@cs.uu.nl>
*		- Simon Goodall <sg02r@ecs.soton.ac.uk> 
*		- Fixed a output bugs using getType() instead of getTypeName().
* 
******************************************************************/

#include <x3d/ShapeNode.h>

using namespace CyberX3D;

static const char appearanceExposedFieldName[] = "appearance";
static const char geometryExposedFieldName[] = "geometry";

ShapeNode::ShapeNode() 
{
	setHeaderFlag(false);
	setType(SHAPE_NODE);

	// appearance field
	appField = new SFNode();
	addExposedField(appearanceExposedFieldName, appField);

	// geometry field
	geomField = new SFNode();
	addExposedField(geometryExposedFieldName, geomField);
}

ShapeNode::~ShapeNode() 
{
}

////////////////////////////////////////////////
//	Appearance
////////////////////////////////////////////////

SFNode *ShapeNode::getAppearanceField() const
{
	if (isInstanceNode() == false)
		return appField;
	return (SFNode *)getExposedField(appearanceExposedFieldName);
}

////////////////////////////////////////////////
//	Geometry
////////////////////////////////////////////////

SFNode *ShapeNode::getGeometryField() const
{
	if (isInstanceNode() == false)
		return geomField;
	return (SFNode *)getExposedField(geometryExposedFieldName);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

ShapeNode *ShapeNode::next() const
{
	return (ShapeNode *)Node::next(getType());
}

ShapeNode *ShapeNode::nextTraversal() const
{
	return (ShapeNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	Geometry
////////////////////////////////////////////////

Geometry3DNode *ShapeNode::getGeometry3D() const
{
	for (Node *node=getChildNodes(); node; node=node->next()) {
		if (node->isGeometry3DNode())
			return (Geometry3DNode *)node;
	}
	return NULL;
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool ShapeNode::isChildNodeType(Node *node) const
{
	if (node->isAppearanceNode() || node->isGeometry3DNode())
		return true;
	else
		return false;
}

void ShapeNode::initialize() 
{
}

void ShapeNode::uninitialize() 
{
}

void ShapeNode::update() 
{
}

////////////////////////////////////////////////
//	Infomation
////////////////////////////////////////////////

void ShapeNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	AppearanceNode *appearance = getAppearanceNodes();
	if (appearance != NULL) {
		if (appearance->isInstanceNode() == false) {
			if (appearance->getName() != NULL && strlen(appearance->getName()))
				printStream << indentString << "\t" << "appearance " << "DEF " << appearance->getName() << " Appearance {" << std::endl;
			else
				printStream << indentString << "\t" << "appearance Appearance {" << std::endl;
			appearance->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "appearance USE " << appearance->getName() << std::endl;
	}
	
	Node *node = getGeometry3DNode();
	if (node != NULL) {
		if (node->isInstanceNode() == false) {
			if (node->getName() != NULL && strlen(node->getName()))
				printStream << indentString << "\t" << "geometry " << "DEF " << node->getName() << " " << node->Node::getTypeString() << " {" << std::endl;
			else
				printStream << indentString << "\t" << "geometry " << node->getTypeString() << " {" << std::endl;
			node->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "geometry USE " << node->getName() << std::endl;
	}
}

