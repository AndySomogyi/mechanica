/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	AppearanceNode.cpp
*
*	Revisions:
*
*	12/01/02
*		- Added the follwing new X3D fields.
*			lineProperties,  fillProperties
*	12/05/03
*		- Shenyang <shenyang@163.net>
*		- Dennis <dennis@cs.uu.nl>
*		- Simon Goodall <sg02r@ecs.soton.ac.uk> 
*		- Fixed a output bugs using getType() instead of getTypeName().
*
******************************************************************/

#include <x3d/AppearanceNode.h>
#include <x3d/TextureNode.h>

using namespace CyberX3D;

static const char materialExposedFieldName[] = "material";
static const char textureExposedFieldName[] = "texture";
static const char textureTransformExposedFieldName[] = "textureTransform";
static const char linePropertiesExposedFieldName[] = "lineProperties";
static const char fillPropertiesExposedFieldName[] = "fillProperties";

AppearanceNode::AppearanceNode() 
{
	setHeaderFlag(false);
	setType(APPEARANCE_NODE);

	///////////////////////////
	// VRML97 Field 
	///////////////////////////
		
	materialField = new SFNode();
	addExposedField(materialExposedFieldName, materialField);

	textureField = new SFNode();
	addExposedField(textureExposedFieldName, textureField);

	texTransformField = new SFNode();
	addExposedField(textureTransformExposedFieldName, texTransformField);

	///////////////////////////
	// X3D Field 
	///////////////////////////
		
	linePropertiesField = new SFNode();
	addExposedField(linePropertiesExposedFieldName, linePropertiesField);

	fillPropertiesField = new SFNode();
	addExposedField(fillPropertiesExposedFieldName, fillPropertiesField);
}

AppearanceNode::~AppearanceNode() 
{
}

////////////////////////////////////////////////
//	SFNodes
////////////////////////////////////////////////

SFNode *AppearanceNode::getMaterialField() const
{
	if (isInstanceNode() == false)
		return materialField;
	return (SFNode *)getExposedField(materialExposedFieldName);
}

SFNode *AppearanceNode::getTextureField() const
{
	if (isInstanceNode() == false)
		return textureField;
	return (SFNode *)getExposedField(textureExposedFieldName);
}

SFNode *AppearanceNode::getTextureTransformField() const 
{
	if (isInstanceNode() == false)
		return texTransformField;
	return (SFNode *)getExposedField(textureTransformExposedFieldName);
}

SFNode *AppearanceNode::getLinePropertiesField() const 
{
	if (isInstanceNode() == false)
		return linePropertiesField;
	return (SFNode *)getExposedField(linePropertiesExposedFieldName);
}

SFNode *AppearanceNode::getFillPropertiesField() const 
{
	if (isInstanceNode() == false)
		return fillPropertiesField;
	return (SFNode *)getExposedField(fillPropertiesExposedFieldName);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

AppearanceNode *AppearanceNode::next() const 
{
	return (AppearanceNode *)Node::next(getType());
}

AppearanceNode *AppearanceNode::nextTraversal() const 
{
	return (AppearanceNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	virtual functions
////////////////////////////////////////////////
	
bool AppearanceNode::isChildNodeType(Node *node) const
{
	if (node->isMaterialNode() || node->isTextureNode() || node->isTextureTransformNode())
		return true;
	else
		return false;
}

void AppearanceNode::initialize() 
{
}

void AppearanceNode::uninitialize() 
{
}

void AppearanceNode::update() 
{
}

void AppearanceNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	MaterialNode *material = getMaterialNodes();
	if (material != NULL) {
		if (material->isInstanceNode() == false) {
			if (material->getName() != NULL && strlen(material->getName()))
				printStream << indentString << "\t" << "material " << "DEF " << material->getName() << " Material {" << std::endl;
			else
				printStream << indentString << "\t" << "material Material {" << std::endl;
			material->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "material USE " << material->getName() << std::endl;
	}

	TextureNode *texture = getTextureNode();
	if (texture != NULL) {
		if (texture->isInstanceNode() == false) {
			if (texture->getName() != NULL && strlen(texture->getName()))
				printStream << indentString << "\t" << "texture " << "DEF " << texture->getName() << " " << texture->Node::getTypeString() << " {" << std::endl;
			else
				printStream << indentString << "\t" << "texture " << texture->Node::getTypeString() << " {" << std::endl;
			texture->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "texture USE " << texture->getName() << std::endl;
	}

	TextureTransformNode *textureTransform = getTextureTransformNodes();
	if (textureTransform != NULL) {
		if (textureTransform->isInstanceNode() == false) {
			if (textureTransform->getName() != NULL && strlen(textureTransform->getName()))
				printStream << indentString << "\t" << "textureTransform " << "DEF " << textureTransform->getName() << " TextureTransform {" << std::endl;
			else
				printStream << indentString << "\t" << "textureTransform TextureTransform {" << std::endl;
			textureTransform->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "textureTransform USE " << textureTransform->getName() << std::endl;
	}
}
