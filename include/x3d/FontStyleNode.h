/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	FontStyleNode.h
*
******************************************************************/

#ifndef _CX3D_FONTSTYLE_H_
#define _CX3D_FONTSTYLE_H_

#include <x3d/VRML97Fields.h>
#include <x3d/Node.h>

namespace CyberX3D {

enum {
FONTSTYLE_FAMILY_SERIF,
FONTSTYLE_FAMILY_SANS,
FONTSTYLE_FAMILY_TYPEWRITER,
};

enum {
FONTSTYLE_STYLE_PLAIN,
FONTSTYLE_STYLE_BOLD,
FONTSTYLE_STYLE_ITALIC,
FONTSTYLE_STYLE_BOLDITALIC,
};

enum {
FONTSTYLE_JUSTIFY_BEGIN,
FONTSTYLE_JUSTIFY_MIDDLE,
FONTSTYLE_JUSTIFY_END,
FONTSTYLE_JUSTIFY_FIRST,
};

class FontStyleNode : public Node {

	SFString *familyField;
	SFString *styleField;
	SFString *languageField;
	MFString *justifyField;
	SFFloat *sizeField;
	SFFloat *spacingField;
	SFBool *horizontalField;
	SFBool *leftToRightField;
	SFBool *topToBottomField;
	
public:

	FontStyleNode();
	virtual ~FontStyleNode();

	////////////////////////////////////////////////
	//	Size
	////////////////////////////////////////////////

	SFFloat *getSizeField() const;

	void setSize(float value);
	float getSize() const;

	////////////////////////////////////////////////
	//	Family
	////////////////////////////////////////////////
	
	SFString *getFamilyField() const;

	void setFamily(const char *value);
	const char *getFamily() const;
	int getFamilyNumber() const;

	////////////////////////////////////////////////
	//	Style
	////////////////////////////////////////////////
	
	SFString *getStyleField() const;

	void setStyle(const char *value);
	const char *getStyle() const;
	int getStyleNumber() const;

	////////////////////////////////////////////////
	//	Language
	////////////////////////////////////////////////
	
	SFString *getLanguageField() const;

	void setLanguage(const char *value);
	const char *getLanguage() const;

	////////////////////////////////////////////////
	//	Horizontal
	////////////////////////////////////////////////
	
	SFBool *getHorizontalField() const;

	void setHorizontal(bool value);
	void setHorizontal(int value);
	bool getHorizontal() const;

	////////////////////////////////////////////////
	//	LeftToRight
	////////////////////////////////////////////////
	
	SFBool *getLeftToRightField() const;

	void setLeftToRight(bool value);
	void setLeftToRight(int value);
	bool getLeftToRight() const;

	////////////////////////////////////////////////
	//	TopToBottom
	////////////////////////////////////////////////
	
	SFBool *getTopToBottomField() const;

	void setTopToBottom(bool value);
	void setTopToBottom(int value);
	bool getTopToBottom() const;

	////////////////////////////////////////////////
	// Justify
	////////////////////////////////////////////////

	MFString *getJustifyField() const;

	void addJustify(const char *value);
	int getNJustifys() const;
	const char *getJustify(int index) const;

	////////////////////////////////////////////////
	//	Spacing
	////////////////////////////////////////////////

	SFFloat *getSpacingField() const;

	void setSpacing(float value);
	float getSpacing() const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	FontStyleNode *next() const;
	FontStyleNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	Justifymation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

