/******************************************************************
*
*	CyberX3D for C++
*
*	Copyright (C) Satoshi Konno 1996-2007
*
*	File:	MultiTextureNodeNode.h
*
******************************************************************/

#ifndef _CX3D_MULTITEXTURENODE_H_
#define _CX3D_MULTITEXTURENODE_H_

#include <x3d/X3DFields.h>
#include <x3d/TextureNode.h>

namespace CyberX3D {

class MultiTextureNode : public TextureNode
{
	SFBool *materialColorField;
	SFBool *materialAlphaField;
	SFBool *transparentField;
	SFBool *nomipmapField;
	MFString *modeField;
	SFNode *textureField;
	SFNode *texTransformField;
	SFColor *colorField;
	SFFloat *alphaField;

public:

	MultiTextureNode();
	virtual ~MultiTextureNode(); 

	////////////////////////////////////////////////
	//	SFNode Field
	////////////////////////////////////////////////

	SFNode *getTextureField() const;
	SFNode *getTextureTransformField() const;

	////////////////////////////////////////////////
	//	MaterialColor
	////////////////////////////////////////////////

	SFBool *getMaterialColorField() const;

	void setMaterialColor(bool value);
	bool getMaterialColor() const;
	bool isMaterialColor() const;

	////////////////////////////////////////////////
	//	MaterialAlpha
	////////////////////////////////////////////////

	SFBool *getMaterialAlphaField() const;

	void setMaterialAlpha(bool value);
	bool getMaterialAlpha() const;
	bool isMaterialAlpha() const;

	////////////////////////////////////////////////
	//	Transparent
	////////////////////////////////////////////////

	SFBool *getTransparentField() const;

	void setTransparent(bool value);
	bool getTransparent() const;
	bool isTransparent() const;

	////////////////////////////////////////////////
	//	Nomipmap
	////////////////////////////////////////////////

	SFBool *getNomipmapField() const;

	void setNomipmap(bool value);
	bool getNomipmap() const;
	bool isNomipmap() const;

	////////////////////////////////////////////////
	// Mode
	////////////////////////////////////////////////

	MFString *getModeField() const;

	void	addMode(char *value);
	int		getNModes() const;
	const char *getMode(int index) const;

	////////////////////////////////////////////////
	//	Alpha
	////////////////////////////////////////////////

	SFFloat *getAlphaField() const;
	
	void setAlpha(float value);
	float getAlpha() const;

	////////////////////////////////////////////////
	//	Color
	////////////////////////////////////////////////

	SFColor *getColorField() const;

	void setColor(float value[]);
	void setColor(float r, float g, float b);
	void getColor(float value[]) const;

	////////////////////////////////////////////////
	//	List
	////////////////////////////////////////////////

	MultiTextureNode	*next() const;
	MultiTextureNode	*nextTraversal() const;

	////////////////////////////////////////////////
	//	virtual functions
	////////////////////////////////////////////////

	bool	isChildNodeType(Node *node) const;
	void	initialize();
	void	uninitialize();
	void	update();
	void	outputContext(std::ostream &printStream, const char *indentString) const;

};

}

#endif
