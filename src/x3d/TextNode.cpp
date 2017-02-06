/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TextNode.cpp
*
*	Revisions:
*
*	12/04/02
*		- Added new fontStyle field of X3D.
*
******************************************************************/

#include <assert.h>
#include <string.h>

#include <x3d/TextNode.h>
#include <x3d/Graphic3D.h>

using namespace CyberX3D;

static const char fontStyleFieldString[] = "fontStyle";

TextNode::TextNode() 
{
	setHeaderFlag(false);
	setType(TEXT_NODE);

	///////////////////////////
	// ExposedField 
	///////////////////////////

	// maxExtent exposed field
	maxExtentField = new SFFloat(1.0f);
	addExposedField(maxExtentFieldString, maxExtentField);

	// length exposed field
	lengthField = new MFFloat();
	addExposedField(lengthFieldString, lengthField);

	// string exposed field
	stringField = new MFString();
	addExposedField(stringFieldString, stringField);

	// string exposed field
	fontStyleField = new SFNode();
	addExposedField(fontStyleFieldString, fontStyleField);
}

TextNode::~TextNode() 
{
}

////////////////////////////////////////////////
//	FontStyle
////////////////////////////////////////////////

SFNode *TextNode::getFontStyleField() 
{
	if (isInstanceNode() == false)
		return fontStyleField;
	return (SFNode *)getExposedField(fontStyleFieldString);
}

////////////////////////////////////////////////
//	MaxExtent
////////////////////////////////////////////////

SFFloat *TextNode::getMaxExtentField() const
{
	if (isInstanceNode() == false)
		return maxExtentField;
	return (SFFloat *)getExposedField(maxExtentFieldString);
}
	
void TextNode::setMaxExtent(float value) 
{
	getMaxExtentField()->setValue(value);
}

float TextNode::getMaxExtent() const
{
	return getMaxExtentField()->getValue();
} 

////////////////////////////////////////////////
// String
////////////////////////////////////////////////

MFString *TextNode::getStringField() const
{
	if (isInstanceNode() == false)
		return stringField;
	return (MFString *)getExposedField(stringFieldString);
}

void TextNode::addString(const char *value) 
{
	getStringField()->addValue(value);
}

int TextNode::getNStrings() const
{
	return getStringField()->getSize();
}

const char *TextNode::getString(int index) const
{
	return getStringField()->get1Value(index);
}

void TextNode::setString(int index, const char* value) 
{
	getStringField()->set1Value(index, value);
}

////////////////////////////////////////////////
// length
////////////////////////////////////////////////

MFFloat *TextNode::getLengthField() const
{
	if (isInstanceNode() == false)
		return lengthField;
	return (MFFloat *)getExposedField(lengthFieldString);
}

void TextNode::addLength(float value) 
{
	getLengthField()->addValue(value);
}

int TextNode::getNLengths() const
{
	return getLengthField()->getSize();
}

float TextNode::getLength(int index) const
{
	return getLengthField()->get1Value(index);
}

////////////////////////////////////////////////
//	List
////////////////////////////////////////////////

TextNode *TextNode::next() const
{
	return (TextNode *)Node::next(getType());
}

TextNode *TextNode::nextTraversal() const
{
	return (TextNode *)Node::nextTraversalByType(getType());
}

////////////////////////////////////////////////
//	functions
////////////////////////////////////////////////
	
bool TextNode::isChildNodeType(Node *node) const
{
	if (node->isFontStyleNode())
		return true;
	else
		return false;
}

void TextNode::initialize() 
{
	recomputeBoundingBox();
#ifdef CX3D_SUPPORT_OPENGL
	recomputeDisplayList();
#endif
}

void TextNode::uninitialize() 
{
}

void TextNode::update() 
{
}

////////////////////////////////////////////////
//	FontStyle
////////////////////////////////////////////////

int TextNode::getFontStyleFamilyNumber() const
{
	FontStyleNode *fontStyle = getFontStyleNodes();
	
	if (fontStyle == NULL)
		return FONTSTYLE_FAMILY_SERIF;
	return fontStyle->getFamilyNumber();
}

int TextNode::getFontStyleStyleNumber() const
{
	FontStyleNode *fontStyle = getFontStyleNodes();
		
	if (fontStyle == NULL)
		return FONTSTYLE_STYLE_PLAIN;

	return fontStyle->getStyleNumber();
}

////////////////////////////////////////////////
//	Stringmation
////////////////////////////////////////////////

void TextNode::outputContext(std::ostream &printStream, const char *indentString) const
{
	printStream << indentString << "\t" << "maxExtent " << getMaxExtent() << std::endl;

	if (0 < getNStrings()) {
		MFString *string = getStringField();
		printStream << indentString << "\t" << "string [" << std::endl;
		string->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]"<< std::endl;
	}

	if (0 < getNLengths()) {
		MFFloat *length = getLengthField();
		printStream << indentString << "\t" << "length [" << std::endl;
		length->MField::outputContext(printStream, indentString, "\t\t");
		printStream << indentString << "\t" << "]"<< std::endl;
	}

	FontStyleNode *fontStyle = getFontStyleNodes();
	if (fontStyle != NULL) {
		if (fontStyle->isInstanceNode() == false) {
			if (fontStyle->getName() != NULL && strlen(fontStyle->getName()))
				printStream << indentString << "\t" << "fontStyle " << "DEF " << fontStyle->getName() << " FontStyle {" << std::endl;
			else
				printStream << indentString << "\t" << "fontStyle FontStyle {"<< std::endl;
			fontStyle->Node::outputContext(printStream, indentString, "\t");
			printStream << indentString << "\t" << "}" << std::endl;
		}
		else 
			printStream << indentString << "\t" << "fontStyle USE " << fontStyle->getName() << std::endl;
	}
}

////////////////////////////////////////////////
//	TextNode::recomputeBoundingBox
////////////////////////////////////////////////

void TextNode::recomputeBoundingBox() 
{
	int nStrings = getNStrings();
	const char *string = NULL;
	if (0 < nStrings) {
		string = getString(0);
		if (string != NULL) {
			if (strlen(string) <= 0)
				string = NULL;
		}
	}
	
	if (string != NULL) {
		float width = (float)strlen(string);
		setBoundingBoxCenter(-width/4.0f/1.0f, 0.5f, 0.0f);
		setBoundingBoxSize(width/4.0f, 0.5f, 0.5f);
	}
	else {
		setBoundingBoxCenter(0.0f, 0.0f, 0.0f);
		setBoundingBoxSize(-1.0f, -1.0f, -1.0f);
	}
}

////////////////////////////////////////////////
//	CX3D_SUPPORT_OPENGL
////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_OPENGL) && defined(WIN32)

OGLFontOutline *TextNode::getOGLFontOutlines() 
{
	return mOGLFontOutlines.getNodes();
}

OGLFontOutline *TextNode::getOGLFontOutline(int family, int style) 
{
	for (OGLFontOutline *node = getOGLFontOutlines(); node != NULL; node = node->next()) {
		if (family == node->getFamily() && style == node->getStyle())
			return node;
	}
	return NULL;
}

void TextNode::addOGLFontOutline(OGLFontOutline *node) 
{
	mOGLFontOutlines.addNode(node); 
}

void TextNode::addOGLFontOutline(int family, int style, unsigned int id) 
{
	addOGLFontOutline(new OGLFontOutline(family, style, id));
}

int TextNode::getNOGLFontOutlines() 
{
	return mOGLFontOutlines.getNNodes();
}

#endif

////////////////////////////////////////////////
//	TextNode::createUseFontOutline
////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_OPENGL) && defined(WIN32)

unsigned int TextNode::createUseFontOutline(int family, int style)
{
	char *fontName = NULL;
	switch (family) {
	case FONTSTYLE_FAMILY_SERIF:
		fontName ="Times New Roman";
		break;
	case FONTSTYLE_FAMILY_SANS:
		fontName ="Helvetica";
		break;
	case FONTSTYLE_FAMILY_TYPEWRITER:
		fontName ="Courier";
		break;
	}

	assert(fontName != NULL);

	unsigned int id = 0;

#if !defined(CX3D_CX3D_SUPPORT_GLUT)
	LOGFONT lf;

	lf.lfHeight = -MulDiv(12, 96, 72);
	lf.lfWidth = 0;
	lf.lfEscapement = 0;
	lf.lfOrientation = 0;
	lf.lfWeight = (style == FONTSTYLE_STYLE_BOLD || style == FONTSTYLE_STYLE_BOLDITALIC)? 700 : 400;
	lf.lfItalic = (style == FONTSTYLE_STYLE_ITALIC || style == FONTSTYLE_STYLE_BOLDITALIC) ? TRUE : FALSE;
	lf.lfUnderline = FALSE;
	lf.lfStrikeOut = FALSE;
	lf.lfCharSet = ANSI_CHARSET;
	lf.lfOutPrecision = OUT_DEFAULT_PRECIS;
	lf.lfClipPrecision = CLIP_DEFAULT_PRECIS;
	lf.lfQuality = DEFAULT_QUALITY;
	lf.lfPitchAndFamily = FF_DONTCARE|DEFAULT_PITCH;
	strcpy(lf.lfFaceName, fontName);

	HFONT font = CreateFontIndirect(&lf);
	HDC hdc = wglGetCurrentDC();

	HFONT oldFont = (HFONT)SelectObject(hdc, font);		

	id = glGenLists(256);
	GLYPHMETRICSFLOAT gmf[256];
	wglUseFontOutlines(hdc, 0, 255, id, 1.0f, 0.1f, WGL_FONT_POLYGONS, gmf);

	SelectObject(hdc, oldFont);		
#endif

	return id;
}

#endif

////////////////////////////////////////////////
//	TextNode::draw
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void TextNode::draw()
{
	unsigned int nDisplayList = getDisplayList();
	if (nDisplayList == 0)
		return;

	int nStrings = getNStrings();
	const char *string = NULL;
	if (0 < nStrings) {
		string = getString(0);
		if (string != NULL) {
			if (strlen(string) <= 0)
				string = NULL;
		}
	}

	if (string == NULL)
		return;

	glListBase(nDisplayList);
	glCallLists(strlen(string), GL_UNSIGNED_BYTE, (const GLvoid*)string);
}

#endif

////////////////////////////////////////////////
//	PointSet::recomputeDisplayList
////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL

void TextNode::recomputeDisplayList() 
{
#ifdef WIN32
	int family	= getFontStyleFamilyNumber();
	int style	= getFontStyleStyleNumber();

	OGLFontOutline *fontOutline = getOGLFontOutline(family, style);
	
	unsigned int id = 0;

	if (fontOutline != NULL) {
		id = fontOutline->getListBaseID();
	}
	else {
		id = createUseFontOutline(family, style);
		addOGLFontOutline(family, style, id);
	}

	//assert(id != 0);

	setDisplayList(id);
#endif
}	

#endif

////////////////////////////////////////////////
//	OGLFontOutline
////////////////////////////////////////////////

#if defined(CX3D_SUPPORT_OPENGL) && defined(WIN32)

LinkedList<OGLFontOutline> TextNode::mOGLFontOutlines;

OGLFontOutline::OGLFontOutline(int family, int style, unsigned int id) 
{
	setFamily(family);
	setStyle(style);
	setListBaseID(id);
}

void OGLFontOutline::setFamily(int family) 
{
	mFamily = family;
}

int OGLFontOutline::getFamily() 
{
	return mFamily;
}

void OGLFontOutline::setStyle(int style) 
{
	mStyle = style;
}

int OGLFontOutline::getStyle() 
{
	return mStyle;
}

void OGLFontOutline::setListBaseID(unsigned int id) 
{
	mListBaseID = id;
}

int OGLFontOutline::getListBaseID() 
{
	return mListBaseID;
}

OGLFontOutline *OGLFontOutline::next() 
{
	return LinkedListNode<OGLFontOutline>::next(); 
}

#endif
