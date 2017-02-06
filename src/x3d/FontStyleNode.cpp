/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FontStyleNode.cpp
*
******************************************************************/

#include <string.h>
#include <x3d/FontStyleNode.h>

using namespace CyberX3D;

FontStyleNode::FontStyleNode() 
{
	setHeaderFlag(false);
	setType(FONTSTYLE_NODE);

	///////////////////////////
	// Field 
	///////////////////////////

	// family field
	familyField = new SFString("SERIF");
	addField(familyFieldString, familyField);

	// style field
	styleField = new SFString("PLAIN");
	addField(styleFieldString, styleField);

	// language field
	languageField = new SFString("");
	addField(languageFieldString, languageField);

	// justify field
	justifyField = new MFString();
	addField(justifyFieldString, justifyField);

	// size field
	sizeField = new SFFloat(1.0f);
	addField(sizeFieldString, sizeField);

	// spacing field
	spacingField = new SFFloat(1.0f);
	addField(spacingFieldString, spacingField);

	// horizontal field
	horizontalField = new SFBool(true);
	addField(horizontalFieldString, horizontalField);

	// leftToRight field
	leftToRightField = new SFBool(true);
	addField(leftToRightFieldString, leftToRightField);

	// topToBottom field
	topToBottomField = new SFBool(true);
	addField(topToBottomFieldString, topToBottomField);
}

FontStyleNode::~FontStyleNode() 
{
}

////////////////////////////////////////////////
//	Size
////////////////////////////////////////////////

SFFloat *FontStyleNode::getSizeField() const
{
	if (isInstanceNode() == false)
		return sizeField;
	return (SFFloat *)getField(sizeFieldString);
}

void FontStyleNode::setSize(float value) 
{
	getSizeField()->setValue(value);
}

float FontStyleNode::getSize() const
{
	return getSizeField()->getValue();
}

////////////////////////////////////////////////
//	Family
////////////////////////////////////////////////

SFString *FontStyleNode::getFamilyField() const
{
	if (isInstanceNode() == false)
		return familyField;
	return (SFString *)getField(familyFieldString);
}
	
void FontStyleNode::setFamily(const char *value) 
{
	getFamilyField()->setValue(value);
}

const char *FontStyleNode::getFamily() const
{
	return getFamilyField()->getValue();
}

////////////////////////////////////////////////
//	Style
////////////////////////////////////////////////

SFString *FontStyleNode::getStyleField() const
{
	if (isInstanceNode() == false)
		return styleField;
	return (SFString *)getField(styleFieldString);
}
	
void FontStyleNode::setStyle(const char *value) 
{
	getStyleField()->setValue(value);
}

const char *FontStyleNode::getStyle() const
{
	return getStyleField()->getValue();
}

////////////////////////////////////////////////
//	Language
////////////////////////////////////////////////

SFString *FontStyleNode::getLanguageField() const
{
	if (isInstanceNode() == false)
		return languageField;
	return (SFString *)getField(languageFieldString);
}
	
void FontStyleNode::setLanguage(const char *value) 
{
	getLanguageField()->setValue(value);
}

const char *FontStyleNode::getLanguage() const
{
	return getLanguageField()->getValue();
}

////////////////////////////////////////////////
//	Horizontal
////////////////////////////////////////////////

SFBool *FontStyleNode::getHorizontalField() const
{
	if (isInstanceNode() == false)
		return horizontalField;
	return (SFBool *)getField(horizontalFieldString);
}
	
void FontStyleNode::setHorizontal(bool value) 
{
	getHorizontalField()->setValue(value);
}

void FontStyleNode::setHorizontal(int value) 
{
	setHorizontal(value ? true : false);
}

bool FontStyleNode::getHorizontal() const
{
	return getHorizontalField()->getValue();
}

////////////////////////////////////////////////
//	LeftToRight
////////////////////////////////////////////////

SFBool *FontStyleNode::getLeftToRightField() const
{
	if (isInstanceNode() == false)
		return leftToRightField;
	return (SFBool *)getField(leftToRightFieldString);
}
	
void FontStyleNode::setLeftToRight(bool value) 
{
	getLeftToRightField()->setValue(value);
}

void FontStyleNode::setLeftToRight(int value) 
{
	setLeftToRight(value ? true : false);
}

bool FontStyleNode::getLeftToRight() const
{
	return getLeftToRightField()->getValue();
}

////////////////////////////////////////////////
//	TopToBottom
////////////////////////////////////////////////

SFBool *FontStyleNode::getTopToBottomField() const
{
	if (isInstanceNode() == false)
		return topToBottomField;
	return (SFBool *)getField(topToBottomFieldString);
}
	
void FontStyleNode::setTopToBottom(bool value) 
{
	getTopToBottomField()->setValue(value);
}

void FontStyleNode::setTopToBottom(int value) 
{
	setTopToBottom(value ? true : false);
}

bool FontStyleNode::getTopToBottom() const
{
	return getTopToBottomField()->getValue();
}

////////////////////////////////////////////////
// Justify
////////////////////////////////////////////////

MFString *FontStyleNode::getJustifyField() const
{
	if (isInstanceNode() == false)
		return justifyField;
	return (MFString *)getField(justifyFieldString);
}

void FontStyleNode::addJustify(const char *value) 
{
	getJustifyField()->addValue(value);
}

int FontStyleNode::getNJustifys() const
{
	return getJustifyField()->getSize();
}

const char *FontStyleNode::getJustify(int index) const
{
	return getJustifyField()->get1Value(index);
}

////////////////////////////////////////////////
//	Spacing
////////////////////////////////////////////////

SFFloat *FontStyleNode::getSpacingField() const
{
	if (isInstanceNode() == false)
		return spacingField;
	return (SFFloat *)getField(spacingFieldString);
}

void FontStyleNode::setSpacing(float value) 
{
	getSpacingField()->setValue(value);
}

float FontStyleNode::getSpacing() const
{
	return getSpacingField()->getValue();
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

FontStyleNode *FontStyleNode::next() const
{
	return (FontStyleNode *)Node::next(getType());
}

FontStyleNode *FontStyleNode::nextTraversal() const
{
	return (FontStyleNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool FontStyleNode::isChildNodeType(Node *node) const
{
	return false;
}

void FontStyleNode::initialize() 
{
	Node *parentNode = getParentNode();
	if (parentNode != NULL) {
		if (parentNode->isTextNode())
			parentNode->initialize();
	}
}

void FontStyleNode::uninitialize() 
{
}

void FontStyleNode::update() 
{
}

////////////////////////////////////////////////
//	Justifymation
////////////////////////////////////////////////

void FontStyleNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	const SFString *family = getFamilyField();
	const SFBool *horizontal = getHorizontalField();
	const SFBool *leftToRight = getLeftToRightField();
	const SFBool *topToBottom = getTopToBottomField();
	const SFString *style = getStyleField();
	const SFString *language = getLanguageField();

	printStream << indentString << "\t" << "size " << getSize() << std::endl;
	printStream << indentString << "\t" << "family " << family << std::endl;
	printStream << indentString << "\t" << "style " << style << std::endl;
	printStream << indentString << "\t" << "horizontal " << horizontal << std::endl;
	printStream << indentString << "\t" << "leftToRight " << leftToRight << std::endl;
	printStream << indentString << "\t" << "topToBottom " << topToBottom << std::endl;
	printStream << indentString << "\t" << "language " << language << std::endl;
	printStream << indentString << "\t" << "spacing " << getSpacing() << std::endl;

	if (0 < getNJustifys()) { 
		const MFString *justify = (MFString *)getField(justifyFieldString);
		printStream << indentString << "\t" << "justify [" << std::endl;
		justify->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]" << std::endl;
	}
}

////////////////////////////////////////////////
//	Text::getFamilyNumber
////////////////////////////////////////////////

int FontStyleNode::getFamilyNumber() const
{
	const char *family = getFamily();

	if (family == NULL)
		return FONTSTYLE_FAMILY_SERIF;

	if (strcmp(family, "SERIF") == 0)
		return FONTSTYLE_FAMILY_SERIF;

	if (strcmp(family, "SANS") == 0)
		return FONTSTYLE_FAMILY_SANS;

	if (strcmp(family, "TYPEWRITER") == 0)
		return FONTSTYLE_FAMILY_TYPEWRITER;

	return FONTSTYLE_FAMILY_SERIF;
}

////////////////////////////////////////////////
//	Text::getStyleNumber
////////////////////////////////////////////////

int FontStyleNode::getStyleNumber() const
{
	const char *style = getStyle();

	if (style == NULL)
		return FONTSTYLE_STYLE_PLAIN;

	if (strcmp(style, "PLAIN") == 0)
		return FONTSTYLE_STYLE_PLAIN;

	if (strcmp(style, "BOLD") == 0)
		return FONTSTYLE_STYLE_BOLD;

	if (strcmp(style, "ITALIC") == 0)
		return FONTSTYLE_STYLE_ITALIC;

	if (strcmp(style, "BOLD ITALIC") == 0)
		return FONTSTYLE_STYLE_BOLDITALIC;

	return FONTSTYLE_STYLE_PLAIN;
}

