/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	TextNode.h
*
******************************************************************/

#ifndef _CX3D_TEXTNODE_H_
#define _CX3D_TEXTNODE_H_

#include <x3d/Geometry3DNode.h>
#include <x3d/FontStyleNode.h>

#if defined(CX3D_SUPPORT_OPENGL) && defined(WIN32)

namespace CyberX3D {

class OGLFontOutline : public LinkedListNode<OGLFontOutline> {
private:
	int				mFamily;
	int				mStyle;
	unsigned int	mListBaseID;
public:
	OGLFontOutline(int family, int style, unsigned int id);
	void setFamily(int family);
	int getFamily() const;
	void setStyle(int style);
	int getStyle() const;
	void setListBaseID(unsigned int id);
	int getListBaseID() const;
	OGLFontOutline *next() const;
};

}

#endif

namespace CyberX3D {

class TextNode : public Geometry3DNode {

	SFNode *fontStyleField;
	SFFloat *maxExtentField;
	MFFloat *lengthField;
	MFString *stringField;
	
public:

	TextNode();
	virtual ~TextNode();

	////////////////////////////////////////////////
	//	FontStyle
	////////////////////////////////////////////////

	SFNode *getFontStyleField();

	////////////////////////////////////////////////
	//	MaxExtent
	////////////////////////////////////////////////

	SFFloat *getMaxExtentField() const;
	
	void setMaxExtent(float value);
	float getMaxExtent() const;

	////////////////////////////////////////////////
	// String
	////////////////////////////////////////////////

	MFString *getStringField() const;

	void addString(const char *value);
	int getNStrings() const;
	const char *getString(int index) const;
	void setString(int index, const char* value);

	////////////////////////////////////////////////
	// length
	////////////////////////////////////////////////

	MFFloat *getLengthField() const;

	void addLength(float value);
	int getNLengths() const;
	float getLength(int index) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	TextNode *next() const;
	TextNode *nextTraversal() const;

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////
	
	bool isChildNodeType(Node *node) const;
	void initialize();
	void uninitialize();
	void update();

	////////////////////////////////////////////////
	//	BoundingBox
	////////////////////////////////////////////////

	void recomputeBoundingBox();

	////////////////////////////////////////////////
	//	functions
	////////////////////////////////////////////////

	int getNPolygons() const {
		return 0;
	}

	////////////////////////////////////////////////
	//	recomputeDisplayList
	////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL
	void recomputeDisplayList();
#endif

	////////////////////////////////////////////////
	//	FontStyle
	////////////////////////////////////////////////

	int getFontStyleFamilyNumber() const;
	int getFontStyleStyleNumber() const;

	////////////////////////////////////////////////
	//	CX3D_SUPPORT_OPENGL
	////////////////////////////////////////////////

#ifdef CX3D_SUPPORT_OPENGL
	void draw();
#endif

#if defined(CX3D_SUPPORT_OPENGL) && defined(WIN32)
	static LinkedList<OGLFontOutline>	mOGLFontOutlines;
	OGLFontOutline *getOGLFontOutlines( const);
	OGLFontOutline *getOGLFontOutline(int family, int style) const;
	void addOGLFontOutline(OGLFontOutline *node);
	unsigned int createUseFontOutline(int family, int style);
	void addOGLFontOutline(int family, int style, unsigned int id);
	int getNOGLFontOutlines() const;
#endif

	////////////////////////////////////////////////
	//	Stringmation
	////////////////////////////////////////////////

	void outputContext(std::ostream &printStream, const char *indentString) const;
};

}

#endif

